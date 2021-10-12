FROM public.ecr.aws/lambda/python:3.8

COPY app.py model.h5 ${LAMBDA_TASK_ROOT}

# # For local testing
# COPY ./entry_script.sh /entry_script.sh
# ADD aws-lambda-rie-x86_64 /usr/local/bin/aws-lambda-rie
# ENTRYPOINT [ "/entry_script.sh" ]

COPY requirements.txt .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD ["app.handler"]