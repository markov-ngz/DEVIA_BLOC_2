FROM alpine:latest

RUN mkdir resources

COPY ./model ./resources/model
COPY ./tokenizer ./resources/tokenizer
