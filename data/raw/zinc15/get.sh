#!/bin/sh
#from : http://zinc15.docking.org/tranches/home/
# filters : React=Clean, Purch=In-Stock, Drug-like
# md5 : 8aa86b495fc04262d1ede38dbde2c6a8
MD5=8aa86b495fc04262d1ede38dbde2c6a8
if [ ! -f zinc15.txt ]; then
    mkdir -p raw
    cd raw
    source ../downloader.curl
    cd ..
    cat raw/*.txt > zinc15.txt
fi
MD5_FILE=$(md5sum zinc15.txt|cut -d' ' -f1)
if [ ! "$MD5" = "$MD5_FILE" ]; then
    echo "Wrong md5 - remove the file to redownload";
    exit;
fi
python process.py
