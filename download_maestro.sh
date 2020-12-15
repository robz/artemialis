dataset='maestro-v2.0.0'
test -f $dataset"-midi.zip" || curl "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/"$dataset"-midi.zip" --output $dataset"-midi.zip"
unzip -n $dataset"-midi.zip" > /dev/null
test -f $dataset".csv" || curl "https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/"$dataset".csv" --output $dataset".csv"
  
