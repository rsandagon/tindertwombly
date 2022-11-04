# download models
mkdir app/models;
mkdir app/downloads;
curl https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json > app/downloads/personachat_self_original.json;
curl https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz > app/downloads/gpt_personachat_cache.tar.gz;
curl https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/finetuned_chatbot_gpt.tar.gz > app/models/finetuned_chatbot_gpt.tar.gz;
chmod -R +x models;
tar -xvzf app/models/finetuned_chatbot_gpt.tar.gz
tar -xvzf app/downloads/gpt_personachat_cache.tar.gz