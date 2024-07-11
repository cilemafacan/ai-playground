#!/bin/bash

# setup.sh
script_directory="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

target_directory="/home/server/storage/sd_models"

directories=("hf_cache" "safetensors" "diffusers_format" "loras")

if [ ! -d "$target_directory" ]; then
  echo "Hedef dizin yok. Oluşturuluyor: $target_directory"
  mkdir -p "$target_directory"
else
  echo "Hedef dizin zaten mevcut: $target_directory"
fi

for dir_name in "${directories[@]}"; do
  dir_path="$target_directory/$dir_name"
  if [ ! -d "$dir_path" ]; then
    echo "Dizin yok. Oluşturuluyor: $dir_path"
    mkdir -p "$dir_path"
  else
    echo "Dizin zaten mevcut: $dir_path"
  fi
done

media_path="$script_directory/1_media/input_images"

if [ ! -d "$media_path" ]; then
  echo "Medya dizini yok. Oluşturuluyor: $media_path"
  mkdir -p "$media_path"
else
  echo "Medya dizini zaten mevcut: $media_path"
fi

env_file=".env"
echo "MEDIA_PATH"="$media_path" > "$env_file"
echo "DF_MODELS_PATH=$target_directory/diffusers_format" >> "$env_file"
echo "SF_MODELS_PATH=$target_directory/safetensors" >> "$env_file"
echo "HF_HOME=$target_directory/hf_cache" >> "$env_file"
echo "LR_MODELS_PATH=$target_directory/loras" >> "$env_file"

if [ -e "$script_directory/requirements.txt" ]; then
  echo "requirements.txt dosyası bulundu. Bağımlılıklar kuruluyor..."
  pip install -r "$script_directory/requirements.txt"
else
  echo "Uyarı: requirements.txt dosyası bulunamadı."
fi