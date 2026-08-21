Offline copy of the case-report model.
======================================

Normally you do not need this folder. run_demo.bat downloads the model from
Hugging Face during setup, once, and every later run reuses it.

Use this folder when the machine cannot reach huggingface.co -- an air-gapped
site, or a network that blocks it. Put the model's files here:

    models\Qwen3-VL-2B-Instruct\
        config.json
        model.safetensors
        tokenizer.json
        ... and the rest of the 12 files

engine/config.py checks for models\Qwen3-VL-2B-Instruct\config.json at import
time. If it is there, the demo loads the model from this folder and never
contacts Hugging Face. If it is not, the demo downloads as usual. Nothing to
configure either way.

Where to get the files
----------------------
On a machine that DOES have internet, either:

  * download them from https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
    (Files and versions -> download each file), or

  * run the demo there once and copy the folder out of the cache:

        %USERPROFILE%\.cache\huggingface\hub\
            models--Qwen--Qwen3-VL-2B-Instruct\snapshots\<hash>\

    Copy the CONTENTS of that snapshot folder, not the folder itself. On
    Windows some of those entries may be links into a sibling "blobs"
    directory -- copy with something that follows links (Explorer copy/paste
    does), so you end up with real files here.

About 4 GB. The model is Qwen/Qwen3-VL-2B-Instruct, Apache-2.0, public: no
Hugging Face account and no token are needed to download it.

Nothing in this folder is committed to git.
