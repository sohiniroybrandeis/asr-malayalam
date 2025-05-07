### Project

This is a research project on ASR (Automatic Speech Recognition). The question we are asking is, can supplementing a lower resource target language with typologically-similar “donor data” from a higher-resource language improve performance in ASR?
The target language in question is the lower-resourced Dravidian language Malayalam. I have experimented with donor languages Telugu, Tamil, and Kannada in addition to Malayalam on wav2vec XLSR-128. Further information can be found within the attached presentation.

### Code Structure

Code is organized into the given files:

- atds.py
  - All code to calculate ATDS (Acoustic Token Distribution Similarity) between two languages.
- cpt_baseline.py
  - All code to pretrain and finetune a model with 10 hours Malayalam.
- cpt_fiftyhour.py
  - All code to pretrain and finetune a model with 30 hours Malayalam + 20 hours Kannada.
- cpt_kanmal.py
  - All code to pretrain and finetune a model with 10 hours Malayalam + 20 hours Kannada.
- cpt_tamal.py
  - All code to pretrain and finetune a model with 10 hours Malayalam + 20 hours Tamil.
- cpt_telgmal.py
  - All code to pretrain and finetune a model with 10 hours Malayalam + 20 hours Telugu.
- cpt_topline.py
  - All code to pretrain and finetune a model with 30 hours Malayalam.
- create_finetune_split.py
  - All code to create 3 hour subset of data for finetuning.
- demo.py
  - All code to run a pretrained+finetuned model on a single audio file, to produce predicted transcription.
- new_audio.py
  - All code to convert audio to a HuggingFace dataset.
- unadapted_xlsr.py
  - All code to finetune a model with 3 hours of Malayalam.

Any of the files can be run with `python *file_name.py*`
