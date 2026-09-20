# backend/lambda/analyze/Dockerfile
#
# Container image for the sentrisense-inference Lambda function.
# Switched from ZIP deployment to a container image because the combined
# dependency set (onnxruntime + transformers + scikit-learn) is too large
# to fit Lambda's ZIP deployment size limits.
#
# Models are NOT baked into this image — they're loaded from S3 at runtime
# (see MODEL_BUCKET / *_KEY environment variables on the Lambda function),
# so this image only needs the code and its Python dependencies.

FROM public.ecr.aws/lambda/python:3.13

# Install only the runtime dependencies this Lambda actually needs — a
# trimmed-down subset of the full project requirements.txt, not the whole
# ML training/experimentation stack (torch, datasets, matplotlib, etc. are
# NOT needed at inference time and would only bloat the image).
COPY backend/lambda/analyze/requirements.txt ${LAMBDA_TASK_ROOT}/requirements.txt
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy the actual Lambda handler code
COPY backend/lambda/analyze/handler.py ${LAMBDA_TASK_ROOT}/handler.py
COPY backend/lambda/analyze/inference.py ${LAMBDA_TASK_ROOT}/inference.py

# Matches the existing Lambda's configured handler: "handler.handler"
CMD ["handler.handler"]
