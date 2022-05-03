#!/bin/bash

set -xev

if [ ! -d "/alpa" ]
then
  echo "/alpa must be checked out and mounted."
  exit 1
fi

cd alpa/docs
make html
