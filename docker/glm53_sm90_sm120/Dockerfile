ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}

ARG PUTPOCKET_SOURCE_COMMIT

LABEL org.opencontainers.image.title="PutPocket GLM-5.3 RunPod SM90+SM120 W4A16" \
      org.opencontainers.image.description="Current vLLM GLM5Next plus default-OFF PutPocket indexer capture and stateful-edit-v3 accuracy ablation" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      org.opencontainers.image.revision="9cd956c7e6cf54efa366b803cafa15ec6c2df827" \
      putpocket.task_id="T20260907-001__glm53-runpod-sm90" \
      putpocket.target.cuda.arch="9.0a,12.0a" \
      putpocket.model.id="Intel/GLM-5.3-Flash-W4A16-AutoRound" \
      putpocket.model.revision="5eee1846f0321058ed73745f9aa16f2aaf0fc0a0" \
      putpocket.model.weights_in_image="false" \
      putpocket.source.commit="${PUTPOCKET_SOURCE_COMMIT}" \
      putpocket.runtime.gpu_validated="false" \
      putpocket.glm53.stateful_edit_default="off" \
      putpocket.glm53.stateful_edit_semantics="full-target-prefill-then-donor-overwrite-accuracy-ablation" \
      putpocket.glm53.true_partial_prefill="false"

WORKDIR /opt/putpocket
RUN mkdir -p /opt/putpocket/src/putpocket_dataset_mining \
      /opt/putpocket/configs/models \
      /opt/putpocket/configs/execution \
      /opt/putpocket/configs/experiments/schemas \
      /opt/putpocket/instrumentation/vllm \
      /opt/putpocket/scripts/glm53_runpod \
      /opt/putpocket/docs \
      /models /data /results

COPY src/putpocket_dataset_mining/glm53_runpod_image.py src/putpocket_dataset_mining/glm53_runpod_stateful_edit.py src/putpocket_dataset_mining/glm53_runpod_stateful_proxy.py /opt/putpocket/src/putpocket_dataset_mining/
COPY instrumentation/vllm/glm53_runpod_stateful_edit_accuracy_ablation.py /opt/putpocket/instrumentation/vllm/glm53_runpod_stateful_edit_accuracy_ablation.py
COPY configs/models/glm53_flash_w4a16_runpod_sm90_sm120.lock.json /opt/putpocket/configs/models/glm53_flash_w4a16_runpod_sm90_sm120.lock.json
COPY configs/execution/glm53_runpod_sm90_sm120_w4a16.example.yaml /opt/putpocket/configs/execution/glm53_runpod_sm90_sm120_w4a16.example.yaml
COPY configs/experiments/schemas/glm53_runpod_stateful_edit_control.schema.json /opt/putpocket/configs/experiments/schemas/glm53_runpod_stateful_edit_control.schema.json
COPY scripts/glm53_runpod /opt/putpocket/scripts/glm53_runpod
COPY patches/vllm/9cd956c7e6cf54efa366b803cafa15ec6c2df827/glm53_sm90_sm120_build.patch patches/vllm/9cd956c7e6cf54efa366b803cafa15ec6c2df827/glm53_nope_fp8_ds_mla_cache.patch patches/vllm/9cd956c7e6cf54efa366b803cafa15ec6c2df827/glm53_sm120_nope_topk_lens.patch patches/vllm/9cd956c7e6cf54efa366b803cafa15ec6c2df827/glm53_stateful_edit_v3_accuracy_ablation.patch /opt/putpocket/patches/vllm/9cd956c7e6cf54efa366b803cafa15ec6c2df827/
COPY docker/glm53_sm90_sm120/Dockerfile /opt/putpocket/docker/glm53_sm90_sm120/Dockerfile
COPY docs/GLM53_RUNPOD_SM90_SM120_IMAGE.md /opt/putpocket/docs/GLM53_RUNPOD_SM90_SM120_IMAGE.md

ENV PYTHONPATH=/opt/putpocket/src \
    MODEL_PATH=/models/glm53-flash-w4a16-autoround \
    MODEL_ID=Intel/GLM-5.3-Flash-W4A16-AutoRound \
    MODEL_REVISION=5eee1846f0321058ed73745f9aa16f2aaf0fc0a0 \
    PUTPOCKET_GLM53_RUNTIME_PROFILE=sm90 \
    PUTPOCKET_GLM53_STATEFUL_EDIT_CONTROL= \
    TENSOR_PARALLEL_SIZE=4 \
    PIPELINE_PARALLEL_SIZE=1 \
    VLLM_PORT=8000 \
    PUTPOCKET_PROXY_PORT=18000 \
    MAX_MODEL_LEN=4096 \
    GPU_MEMORY_UTILIZATION=0.92 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN set -eux; \
    chmod 0755 /opt/putpocket/scripts/glm53_runpod/*.sh /opt/putpocket/scripts/glm53_runpod/doctor.py; \
    python3 -m py_compile \
      /opt/putpocket/src/putpocket_dataset_mining/glm53_runpod_image.py \
      /opt/putpocket/src/putpocket_dataset_mining/glm53_runpod_stateful_edit.py \
      /opt/putpocket/src/putpocket_dataset_mining/glm53_runpod_stateful_proxy.py \
      /opt/putpocket/instrumentation/vllm/glm53_runpod_stateful_edit_accuracy_ablation.py; \
    python3 -c 'import putpocket_dataset_mining.glm53_runpod_image, putpocket_dataset_mining.glm53_runpod_stateful_edit, putpocket_dataset_mining.glm53_runpod_stateful_proxy'

EXPOSE 8000 18000
VOLUME ["/models", "/data", "/results"]
HEALTHCHECK --interval=30s --timeout=20s --start-period=30s --retries=3 CMD ["/opt/putpocket/scripts/glm53_runpod/entrypoint.sh", "doctor"]
ENTRYPOINT ["/opt/putpocket/scripts/glm53_runpod/entrypoint.sh"]
CMD ["doctor"]
