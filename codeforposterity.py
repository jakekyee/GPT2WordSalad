!pip install -q gpt-2-simple
!pip install tensorflow

from google.colab import files
uploaded = files.upload()

import gpt_2_simple as gpt2
from datetime import datetime
from google.colab import files

gpt2.download_gpt2(model_name="124M")

file_name = "testtext.txt"
run_name = 'runforestrun'
model_size = '124M'

sess = gpt2.start_tf_sess()

gpt2.finetune(sess,
              dataset=file_name,
              model_name=model_size,
              steps=200,
              restore_from='fresh',
              run_name = run_name,
              print_every=10,
              sample_every=50,
              save_every=50)