#!/bin/sh
MD5=3c20f9801834b8f4b25b8ad40a6dda1d
if [ ! -f chembl_22_chemreps.txt.gz ]; then
    wget ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_22/archived/chembl_22_chemreps.txt.gz
fi
MD5_FILE=$(md5sum chembl_22_chemreps.txt.gz |cut -d' ' -f1)
if [ ! "$MD5" = "$MD5_FILE" ]; then
    echo "Wrong md5 - remove the file to redownload";
    exit;
fi
python process.py
