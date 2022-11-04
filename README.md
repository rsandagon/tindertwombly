# PROXY WITH SSL #

## FIREWALL CONFIG
* https -> `sudo ufw allow ssh`
* https -> `sudo ufw allow 443/tcp`
* https -> `sudo ufw allow 80/tcp`

## NETWORK
* Since we started twomblyapi and twombly with docker network mynetwork. docker compose must specify external

##  DOCKER RUN
* `./init-letsencrypt.sh`
*  `docker-compose up`

## LET'S ENCRYPT
* https://medium.com/@pentacent/nginx-and-lets-encrypt-with-docker-in-less-than-5-minutes-b4b8a60d3a71