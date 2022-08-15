#!/usr/bin/env bash

function setNumProcessors () {
    # Set the number of processors used for build
    # to be 1 less than are available
    if [[ -f "$(which nproc)" ]]; then
        NPROC="$(nproc)"
    else
        NPROC="$(grep -c '^processor' /proc/cpuinfo)"
    fi
    echo `expr "${NPROC}" - 1`
}

function main() {
    cd /tmp

    git clone https://github.com/xrootd/xrootd.git /tmp/xroot

    mkdir build
    cd build

    printf "\n# cmake /tmp/xroot -DCMAKE_INSTALL_PREFIX=/opt/xrootd\n"
    cmake /tmp/xroot \
        -DCMAKE_INSTALL_PREFIX=/opt/xrootd
    printf "\n# cmake --build . -- -j${NPROC}\n"
    cmake --build . -- -j${NPROC}
    printf "\n# make install\n"
    make install

    rm -rf /tmp/*
}

main "$@" || exit 1
