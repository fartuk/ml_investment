#!/bin/bash

#set -x
set -e

PYTHON=python3.8
PIP="$PYTHON -m pip"
VENV_NAME="ml_investment_venv"
REPO_NAME=ml_investment

# check if being sourced (bash only)
function is_sourced() {
    if [ "$(uname)" == "Darwin" ] ; then
        false
    else
        [[ ${FUNCNAME[-1]} == "source" ]]
    fi
}

is_sourced && set +e && source $VENV_NAME/bin/activate && return

python3.8 --version

if [ "$1" == freeze -o "$1" == f ] ; then
    $PIP freeze -l | grep -v "ml_investment" > requirements.txt
    echo TODO many shitty and uninstallable packages come here
    echo UPDATED requirements.txt
    git diff requirements.txt
fi

if [ "$1" == install -o "$1" == i ] ; then
    $PYTHON -m venv $VENV_NAME
    source $VENV_NAME/bin/activate
    $PYTHON -m pip install --upgrade pip
    $PIP install -r ./requirements.txt
    $PIP install -e ../$REPO_NAME
fi

if [ "$1" == activate -o "$1" == a ] ; then
    if [ "$(uname)" == "Darwin" ] ; then
        set +e && source $VENV_NAME/bin/activate
    else
        echo TYPE in command line to activate venv
        echo ". v"
    fi
fi

if [ "$1" == pip ] ; then
    shift
    $PIP $@
fi

# TODO deactivate | clean | ???
