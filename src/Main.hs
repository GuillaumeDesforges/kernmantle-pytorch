{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE OverloadedLabels #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE QuasiQuotes #-}

module Main where

import Control.Arrow (Arrow)
import Control.Kernmantle.Rope
  ( AnyRopeWith,
    HasKleisli,
    liftKleisliIO,
    loosen,
    perform,
    strand,
    untwine,
    weave',
    (&),
  )
import Control.Monad.IO.Class (MonadIO)
import Path (Dir, Path, Abs, Rel, parseAbsDir, parseRelDir, reldir, toFilePath, (</>))
import System.Directory (getCurrentDirectory, createDirectoryIfMissing)
import System.Process (spawnProcess, CmdSpec (..), CreateProcess (..), createProcess, waitForProcess)
import System.Exit (ExitCode(..))

-- * Binary effects

-- | Manage models
data Models i o where
  -- | Pull a model from the internet
  Pull ::
    String -- ^ github (user/repo)
    -> String -- ^ model name
    -> Models () (Path Abs Dir)

-- | Interpret a Model effect
interpretModels ::
  (Arrow core, MonadIO m, HasKleisli m core) =>
  -- | The effect to intepret
  Models i o ->
  core i o
-- Pull a model from the Web
interpretModels (Pull github name) = liftKleisliIO $ \() -> do
  -- Get current working directory
  cwd <- parseAbsDir =<< getCurrentDirectory
  -- Get pytorch hub store directory
  let storeDir = cwd  </> [reldir|./tmp/pytorch/hub|]
  createDirectoryIfMissing True (toFilePath storeDir)
  -- Run docker
  dockerProcessHandle <- spawnProcess "docker" [
      "run",
      -- bind workspace to _"exported"_ directory
      "-v",
      (toFilePath storeDir) <> ":/workspace",
      -- bind script directory to some input directory
      "-v",
      (toFilePath $ cwd </> [reldir|./data/scripts|]) <> ":/scripts",
      -- Run as current user
      "--user",
      "1000",
      -- Set cwd in docker to _"exported"_ directory
      "--workdir",
      "/workspace",
      -- Use pytorch image
      "pytorch/pytorch",
      -- Pyton command
      "python",
      "/scripts/pull.py",
      github,
      name
    ]
  dockerExitCode <- waitForProcess dockerProcessHandle
  case dockerExitCode of
    ExitSuccess -> putStrLn "Model downloaded"
    ExitFailure _ -> putStrLn "Failed to download model"
  -- Return directory that holds the data
  return storeDir

-- * Run

-- | Some dummy example
exampleProgram :: (MonadIO m) => AnyRopeWith '[ '("models", Models)] '[Arrow, HasKleisli m] () (Path Abs Dir)
exampleProgram = strand #models $ Pull "pytorch/vision:v0.6.0" "inception_v3"

main :: IO ()
main = do
  result <-
    exampleProgram
      & loosen
      & weave' #models interpretModels
      & untwine
      & perform ()
  print result
