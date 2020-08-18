{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE QuasiQuotes #-}

module Main where

import Control.Arrow (Arrow)
import Control.Kernmantle.Rope
  ( AnyRopeWith,
    HasKleisli,
    Writer,
    liftKleisliIO,
    loosen,
    perform,
    runCayley,
    runWriter,
    strand,
    untwine,
    weave',
    writing,
    (&),
    type (~>),
  )
import Control.Monad (forM)
import Control.Monad.IO.Class (MonadIO)
import Path
  ( Abs,
    Dir,
    File,
    Path,
    Rel,
    parent,
    parseAbsDir,
    parseRelDir,
    reldir,
    relfile,
    toFilePath,
    (</>),
  )
import System.Directory (createDirectoryIfMissing, getCurrentDirectory)
import System.Exit (ExitCode (..))
import System.Process
  ( CmdSpec (..),
    CreateProcess (..),
    createProcess,
    spawnProcess,
    waitForProcess,
  )

-- * Binary effects

-- | Represent a Machine Learning model for torch hub
data TorchHubModel
  = TorchHubModel
      String
      -- ^ torch hub GitHub
      String
      -- ^ model name

-- | Represent some data to predict on
type Data = Path Abs File

-- | Represent some prediction
type Prediction = Path Abs File

-- | A "Data Science" effect
data Predict i o where
  -- | Pull a model from the internet
  Predict :: TorchHubModel -> Predict Data Prediction

-- * Interpreters

-- | Interpret the @Predict@ effect
handlePredict ::
  (Arrow core, MonadIO m, HasKleisli m core) =>
  Path Abs Dir ->
  Path Abs Dir ->
  Path Abs Dir ->
  Predict i o ->
  (Writer [TorchHubModel] ~> core) i o
-- Pull a model from the Web
handlePredict hubStoreDir scriptsDir predictionDir (Predict model@(TorchHubModel modelGitHub modelName)) =
  writing [model] $
    liftKleisliIO $ \dataFile -> do
      -- Get path to directory of input data file
      let dataDir = parent dataFile
      -- Get path to file to write prediction in
      let predictionFile = predictionDir </> [relfile|./prediction.txt|]
      -- Make a prediction
      dockerProcessHandle <-
        spawnProcess
          "docker"
          [ "run",
            -- bind torch hub directory
            "-v",
            (toFilePath hubStoreDir) <> ":/hub",
            -- bind script directory
            "-v",
            (toFilePath scriptsDir) <> ":/scripts",
            -- bind data directory
            "-v",
            (toFilePath dataDir) <> ":" <> (toFilePath dataDir),
            -- bind output directory
            "-v",
            (toFilePath predictionDir) <> ":" <> (toFilePath predictionDir),
            -- Run as current user
            "--user",
            "1000",
            -- Set cwd in docker to output directory
            "--workdir",
            (toFilePath predictionDir),
            -- Use pytorch image
            "pytorch-custom",
            -- Pyton command
            "python",
            "/scripts/predict.py",
            "/hub",
            modelGitHub,
            modelName,
            (toFilePath dataFile),
            (toFilePath predictionFile)
          ]
      dockerExitCode <- waitForProcess dockerProcessHandle
      case dockerExitCode of
        ExitSuccess -> putStrLn "Prediction done"
        ExitFailure _ -> putStrLn "Failed to predict"
      return predictionFile

-- * Run

downloadModel :: Path Abs Dir -> Path Abs Dir -> TorchHubModel -> IO Bool
downloadModel hubStoreDir scriptsDir (TorchHubModel modelGitHub modelName) = do
  putStrLn $ "Will download " <> modelName
  -- Download the model
  dockerProcessHandle <-
    spawnProcess
      "docker"
      [ "run",
        -- bind torch hub directory
        "-v",
        (toFilePath hubStoreDir) <> ":/hub",
        -- bind script directory
        "-v",
        (toFilePath $ scriptsDir) <> ":/scripts",
        -- Run as current user
        "--user",
        "1000",
        -- Use pytorch image
        "pytorch-custom",
        -- Pyton command
        "python",
        "/scripts/pull.py",
        "/hub",
        modelGitHub,
        modelName
      ]
  dockerExitCode <- waitForProcess dockerProcessHandle
  case dockerExitCode of
    ExitSuccess -> putStrLn "Model downloaded" >> return True
    ExitFailure _ -> putStrLn "Failed to download model" >> return False

runPipeline pipeline = do
  -- Get current working directory
  cwd <- parseAbsDir =<< getCurrentDirectory
  -- Get path to pytorch hub store directory
  let hubStoreDir = cwd </> [reldir|./tmp/pytorch/hub|]
  createDirectoryIfMissing True (toFilePath hubStoreDir)
  -- Get path to scripts directory
  let scriptsDir = cwd </> [reldir|./data/scripts|]
  -- Get path to prediction output files
  let predictionDir = cwd </> [reldir|./tmp/predictions|]

  -- Weave strands, and do the "load-time" processing of the pipeline
  let (models, runtimePipeline) =
        pipeline
          & loosen
          & weave' #dataScience (handlePredict hubStoreDir scriptsDir predictionDir)
          & untwine
          & runWriter
  downloadSuccesses <- forM models (downloadModel hubStoreDir scriptsDir)
  case all id downloadSuccesses of
    True -> do
      result <- runtimePipeline & perform (cwd </> [relfile|./data/images/dog.jpg|])
      print result
    False -> putStrLn "Failed to download one or more models"

-- Some pipeline that will succeed
pipeline = strand #dataScience $ Predict $ TorchHubModel "pytorch/vision:v0.6.0" "inception_v3"

-- Some pipeline that will fail
failingPipeline = strand #dataScience $ Predict $ TorchHubModel "pytorch/vision:v0.6.0" "rubbish"

main = do
  -- Pipeline that succeeds
  runPipeline pipeline
  -- Pipeline that fails
  runPipeline failingPipeline