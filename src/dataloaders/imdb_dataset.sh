#!/bin/bash
if [ ! -d data/imdb ]; then
  mkdir -p data
  wget https://www.dropbox.com/s/8sz1eu99ixvtsy8/imdb.zip
  unzip imdb.zip -d data/
  rm imdb.zip
fi
