#! /bin/bash

# This script is to convert ondrive share links to directly downloadable links
# https://stackoverflow.com/a/61040150
onelink() { echo -n "$1"|base64|sed "s/=$//;s/\//\_/g;s/\+/\-/g;s/^/https:\/\/api\.onedrive\.com\/v1\.0\/shares\/u\!/;s/$/\/root\/content/"; }


to_convert_array=(
    glnet_4g,https://1drv.ms/u/s!AkBbczdRlZvCyG9DjS_AMqnJQiX1?e=tOkhHP
    glnet_9g,https://1drv.ms/u/s!AkBbczdRlZvCyEGFQ6kfWV-wVZei?e=wWwR8z
    glnet_16g,https://1drv.ms/u/s!AkBbczdRlZvCyjaBxC3h-B8CHWMm?e=GfR0kg
    glnet_4g_tklb,https://1drv.ms/u/s!AkBbczdRlZvCyAWcOdlTUa2J3vsX?e=wE4ch1
    glnet_9g_tklb,https://1drv.ms/u/s!AkBbczdRlZvCynPEcg-fC7ZDCMEt?e=bwufeR
    glnet_stl,https://1drv.ms/u/s!AkBbczdRlZvCyBFaAILEMu_dtbbV?e=9Ghhi3
    glnet_stl_paramslot,https://1drv.ms/u/s!AkBbczdRlZvCyG0eat5zCE9upcQR?e=vmbmo1
)


for to_convert in ${to_convert_array[@]}; do
    IFS=',' read model url <<< "${to_convert}"
    downloadable_url=$( onelink $url )
    echo "'$model': '$downloadable_url',"
done
