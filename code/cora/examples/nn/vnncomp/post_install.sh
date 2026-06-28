#!/bin/bash

echo post_install.sh running..

# -------------------------------------------------------------------------
# SETTINGS

USER_NAME=ubuntu
LICENSE_URL='https://drive.google.com/uc?export=download&id=<file-id>'

MATLAB_RELEASE=2025b
EXISTING_MATLAB_LOCATION=$(dirname $(dirname $(readlink -f $(which matlab))))

# define required products (remove already installed products..)
ADDITIONAL_PRODUCTS="Deep_Learning_Toolbox_Converter_for_ONNX_Model_Format"

CURR_DIR=$(pwd)

# -------------------------------------------------------------------------
# ECHO
echo ${USER_NAME}
echo ${LICENSE_URL}
echo ${EXISTING_MATLAB_LOCATION}
echo ${CURR_DIR}
ls -al
# -------------------------------------------------------------------------
# INITIAL GENERAL INSTALLATION
# check if everything is up to date
# export DEBIAN_FRONTEND=noninteractive \
#     && apt-get update \
#     && apt-get install --no-install-recommends --yes \
#     wget \
#     unzip \
#     ca-certificates \
#     && apt-get clean \
#     && apt-get autoremove \
#     && rm -rf /var/lib/apt/lists/*
	
# -------------------------------------------------------------------------
# MATLAB PACKAGE INSTALLATION
wget -q https://www.mathworks.com/mpm/glnxa64/mpm \
    && chmod +x mpm \
    && ./mpm install \
        --destination=${EXISTING_MATLAB_LOCATION} \
        --release=r${MATLAB_RELEASE} \
        --products ${ADDITIONAL_PRODUCTS}	
	
# -------------------------------------------------------------------------
# CORA INSTALLATION
# download license file
curl --retry 100 --retry-connrefused  -L ${LICENSE_URL} -o license.lic
# copy to license folder and delete other license info
cp -f license.lic "${EXISTING_MATLAB_LOCATION}/licenses"
# run installCORA non-interactively
matlab -nodisplay -r "cd ${CURR_DIR}; addpath(genpath('.')); installCORA(false,true,'${CURR_DIR}/code'); savepath"

# -------------------------------------------------------------------------
# FIX GPU DRIVER ISSUES

# Enable GPU persistence mode (prevents driver unloading)
sudo nvidia-smi -pm 1

# Stop apt from auto-upgrading the NVIDIA driver / kernel out from under a running
# benchmark. An unattended-upgrades run mid-benchmark replaces the driver libraries on
# disk while the already-loaded kernel module stays at the old version -> NVML/CUDA
# "Driver/library version mismatch" -> the GPU disappears (gpuDeviceCount == 0) until the
# instance reboots. The trigger is the apt-daily-upgrade.timer (which runs unattended-upgrade),
# NOT just the unattended-upgrades.service, so disable AND mask both the timers and services.
sudo systemctl disable --now apt-daily.timer apt-daily-upgrade.timer unattended-upgrades.service 2>/dev/null || true
sudo systemctl mask apt-daily.service apt-daily-upgrade.service unattended-upgrades.service 2>/dev/null || true

# Belt-and-suspenders: pin every installed NVIDIA + AWS-kernel package so a manual
# `apt upgrade` can't bump them either. This image uses the *-server and linux-*-aws
# package names (not the -generic names the old hold targeted), so query the installed
# set instead of hard-coding names.
sudo apt-mark hold $(dpkg-query -W -f='${Package}\n' 'nvidia-*' 'libnvidia-*' 'linux-aws*' 'linux-image-aws*' 'linux-headers-aws*' 2>/dev/null) 2>/dev/null || true

# Original holds kept as an extra net: harmless no-ops if these exact package names are not
# installed, but they cover the -generic / non-server variants on non-DLAMI images.
sudo apt-mark hold linux-image-generic linux-headers-generic nvidia-driver-535
sudo systemctl disable unattended-upgrades

# -------------------------------------------------------------------------
# DONE
echo post_install.sh done
