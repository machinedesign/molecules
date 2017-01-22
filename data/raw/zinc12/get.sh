#!/bin/sh
#from http://zinc.docking.org/subsets/clean-drug-like
MD5=80b2490cce2124f472505c46b755f630
if [ ! -f 13_prop.xls ]; then
    wget http://zinc.docking.org/db/bysubset/13/13_prop.xls
fi
MD5_FILE=$(md5sum 13_prop.xls|cut -d' ' -f1)
if [ ! "$MD5" = "$MD5_FILE" ]; then
    echo "Wrong md5 - remove the file to redownload";
    exit;
fi
python process.py
