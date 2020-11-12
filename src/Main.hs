{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE QuasiQuotes #-}
{-# LANGUAGE Arrows #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

module Main where

import Prelude hiding (id, (.))
import Control.Category
import Control.Arrow
import Control.Monad.Trans.Writer
import Data.Functor.Identity
import Data.Hashable
import Control.Kernmantle.Rope
  ( Cayley(..),
    loosen,
    runCayley,
    strand,
    untwine,
    weave',
    (&),
  )
  
import Control.Monad (forM)
import Path
  ( Abs,
    Dir,
    File,
    Path,
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
  ( spawnProcess,
    waitForProcess,
  )
import Data.HashSet as HashSet
import GHC.Generics (Generic)
import Data.List (intercalate)

-- * Binary effects

-- | Represent a Machine Learning model for torch hub
data TorchHubModel
  = TorchHubModel
      String
      -- ^ torch hub GitHub
      String
      -- ^ model name
  deriving (Show, Eq, Generic, Hashable)

-- | Represent some data to predict on
type Data = Path Abs File

-- | Represent some prediction
type Prediction = Path Abs File

-- | How to build a docker image for some "docker run" task. Each different spec
-- will trigger a new build.
data DockerImageSpec = DockerImageSpec { baseImage :: String
                             , extraPipPkgs :: HashSet.HashSet String }
  deriving (Eq, Generic, Hashable)

-- | The name of the image is obtained by hashing the specs
dockerSpecImageName :: DockerImageSpec -> String
dockerSpecImageName spec = "kernmantle-pip-" ++ show (hash spec)

-- | A "Data Science" effect
data Predict i o where
  -- | Pull a model from the internet
  Predict :: TorchHubModel -> Predict Data Prediction

-- | Get some image given some identifier
data GetImage a b where
  GetImage :: GetImage String Data

-- | Collect everything that's gonna be needed by the pipeline
data Requirements = Requirements { reqDockerImageSpecs :: HashSet.HashSet DockerImageSpec
                                 , reqModelNames :: HashSet.HashSet TorchHubModel }
instance Semigroup Requirements where
  Requirements d m <> Requirements d' m' = Requirements (d <> d') (m <> m')
instance Monoid Requirements where
  mempty = Requirements mempty mempty

-- | The target in which to interpret the effects
newtype Core a b = Core (a -> IO b, Requirements)
  deriving (Category, Arrow)
           via Cayley (Writer Requirements) (Kleisli IO)

-- * Interpreters. We make them very monomorphic (constrained to one single Core
-- * type) for simplicity's sake, but they could be rendered more polymorphic.

-- | Interprets @GetImage@ by looking into a folder under the CWD
handleGetImage ::
  Path Abs Dir -> GetImage i o -> Core i o
handleGetImage cwd GetImage = Core
  (\imageId -> do
    imgSubPath <- parseRelFile imageId
    return $ cwd </> [reldir|./data/images|] </> imgSubPath
  ,mempty)

predictImageSpec, downloadImageSpec :: DockerImageSpec
predictImageSpec = DockerImageSpec "pytorch/pytorch" (HashSet.fromList ["scipy"])
downloadImageSpec = predictImageSpec  -- This spec happens to be the same, but
                                      -- it could be a different one

-- | Interpret the @Predict@ effect
handlePredict ::
  Path Abs Dir ->
  Path Abs Dir ->
  Path Abs Dir ->
  Predict i o ->
  Core i o
-- Pull a model from the Web
handlePredict hubStoreDir scriptsDir predictionDir (Predict model@(TorchHubModel modelGitHub modelName)) =
  Core (
    \dataFile -> do
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
            (dockerSpecImageName predictImageSpec),
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
    , Requirements (HashSet.fromList [predictImageSpec]) (HashSet.fromList [model]))

-- | Builds some needed docker images
buildDockerImageSpec :: Path Abs Dir -> DockerImageSpec -> IO ()
buildDockerImageSpec dockerFilesFolder spec@(DockerImageSpec base pipPkgs) = do
  let dockerfileContent = "FROM " ++ base ++"\n\n"++"RUN pip install " ++ intercalate " " (HashSet.toList pipPkgs)
      name = dockerSpecImageName spec
  subfolderName <- parseRelDir name
  let subfolder = dockerFilesFolder </> subfolderName
      dockerfile = subfolder </> [relfile|Dockerfile|]
  createDirectoryIfMissing True (toFilePath subfolder)
  writeFile (toFilePath dockerfile) dockerfileContent
  putStrLn $ "Building " ++ name ++ ":\n" ++ dockerfileContent
  dockerProcessHandle <-
    spawnProcess "docker" ["build", toFilePath subfolder, "-t", name]
  dockerExitCode <- waitForProcess dockerProcessHandle
  case dockerExitCode of
    ExitSuccess -> putStrLn "Build successful"
    ExitFailure _ -> error "Build failed"
  
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
        (dockerSpecImageName downloadImageSpec),
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
  let dockerfilesFolder = cwd </> [reldir|./tmp/docker-builds/|]
  -- Get path to pytorch hub store directory
  let hubStoreDir = cwd </> [reldir|./tmp/pytorch/hub|]
  createDirectoryIfMissing True (toFilePath hubStoreDir)
  -- Get path to scripts directory
  let scriptsDir = cwd </> [reldir|./data/scripts|]
  -- Get path to prediction output files
  let predictionDir = cwd </> [reldir|./tmp/predictions|]

  -- Weave strands, and do the "load-time" processing of the pipeline
  let Core (runtimePipeline, Requirements dockerSpecs models) =
        pipeline
          & loosen
          & weave' #predict (handlePredict hubStoreDir scriptsDir predictionDir)
          & weave' #images (handleGetImage cwd)
          & untwine
  putStrLn $ "Finding docker images locally or building them"
  mapM (buildDockerImageSpec dockerfilesFolder)
       (HashSet.toList $ HashSet.insert downloadImageSpec dockerSpecs)
       -- We add the image needed to download to the mix, as it will be needed
       -- by 'downloadModel'
  putStrLn $ "The pipeline uses the following models: " ++ show models
  downloadSuccesses <- forM (HashSet.toList models) (downloadModel hubStoreDir scriptsDir)
  -- If everything is OK, we actually start the pipeline:
  case all id downloadSuccesses of
    True -> do
      result <- do putStrLn "Starting pipeline"
                   runtimePipeline ()
      print result
    False -> putStrLn "Failed to download one or more models"

-- * The pipeline

pipeline = proc () -> do
  image <- getImage -< "dog.jpg"
  -- Some task that will succeed:
  predictWith "inception_v3" -< image
  -- Some task that will fail, model doesn't exist:
  -- predictWith "rubbish" -< image
  where
    getImage = strand #images GetImage
    predictWith model =
      strand #predict $ Predict $ TorchHubModel "pytorch/vision:v0.6.0" model

-- * Main function

main :: IO ()
main = do
  -- Pipeline that succeeds
  runPipeline pipeline
