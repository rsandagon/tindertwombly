# Tinder Twombly API

> Based on Huggingface's [amazing conversation AI](https://github.com/huggingface/transfer-learning-conv-ai)

## Docker
* `/.build.sh`
* `mkdir apps/downloads` then place there `https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz`
* `docker run -d --name twomblyapi -p 8000:80 --network=mynetwork rsandagon/tindertwombly-api:latest`

## References
[http://arxiv.org/abs/1901.08149](http://arxiv.org/abs/1901.08149):
