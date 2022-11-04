# Tinder Twombly #

> “Sometimes I think I have felt everything I'm ever gonna feel. And from here on out, I'm not gonna feel anything new. Just lesser versions of what I've already felt.”
>
> -- <cite>Theodore Twombly "Her"</cite>

##  DOCKER RUN
* API: `docker run -d --name twomblyapi -p 8000:80 --network=mynetwork rsandagon/tindertwombly-api:latest`
* APP: `docker run -d --name twombly -p 8080:80 --network=mynetwork rsandagon/tindertwombly:latest`

docker run -d --name twomblyTemp -p 80:80 --network=mynetwork rsandagon/tindertwombly:latest

##  HOW TO START TWOMBLY
1. Build twombly and twomblyapi first
    * API: `docker run -d --name twomblyapi -p 8000:80 --network=mynetwork rsandagon/tindertwombly-api:latest`
    * APP: `docker run -d --name twombly -p 8080:80 --network=mynetwork rsandagon/tindertwombly:latest`
1. got to `cd proxy` to run ssl and proxy handling 
1. run `./init-letsencrypt` to make certifications for ssl
1. run `docker-compose up -d`

## LET'S ENCRYPT
* https://medium.com/@pentacent/nginx-and-lets-encrypt-with-docker-in-less-than-5-minutes-b4b8a60d3a71
