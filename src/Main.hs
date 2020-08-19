{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE Arrows #-}
{-# LANGUAGE FlexibleContexts #-}

module Main where

import Control.Arrow
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
    parseRelFile,
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
  deriving (Show)

-- | Represent some data to predict on
type Data = Path Abs File

-- | Represent some prediction
type Prediction = Path Abs File

-- | A "Data Science" effect
data Predict i o where
  -- | Pull a model from the internet
  Predict :: TorchHubModel -> Predict Data Prediction


-- | Get some image given some identifier
data GetImage a b where
  GetImage :: GetImage String Data

-- * Interpreters

-- | Interprets @GetImage@ by looking into a folder under the CWD
handleGetImage :: (Arrow core, MonadIO m, HasKleisli m core) =>
  Path Abs Dir -> GetImage i o -> core i o
handleGetImage cwd GetImage =
  liftKleisliIO $ \imageId -> do
    imgSubPath <- parseRelFile imageId
    return $ cwd </> [reldir|./data/images|] </> imgSubPath

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
            "1001",
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
  putStrLn $ "Downloading model: " <> modelName
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
        "1001",
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
          & weave' #predict (handlePredict hubStoreDir scriptsDir predictionDir)
          & weave' #images (handleGetImage cwd)
          & untwine
          & runWriter
  putStrLn $ "The pipeline uses the following models: " ++ show models
  downloadSuccesses <- forM models (downloadModel hubStoreDir scriptsDir)
  case all id downloadSuccesses of
    True -> do
      result <- do putStrLn "Starting pipeline"
                   runtimePipeline & perform ()
      print result
    False -> putStrLn "Failed to download one or more models"

pipeline = proc () -> do
  image <- getImage -< "dog.jpg"
  -- Some task that will succeed:
  predictWith "inception_v3" -< image
  -- Some task that will fail, model doesn't exist:
  predictWith "rubbish" -< image
  where
    getImage = strand #images GetImage
    predictWith model =
      strand #predict $ Predict $ TorchHubModel "pytorch/vision:v0.6.0" model

main = do
  -- Pipeline that succeeds
  runPipeline pipeline
