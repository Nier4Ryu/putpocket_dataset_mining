ARG VLLM_BASE_IMAGE
FROM ${VLLM_BASE_IMAGE}

LABEL org.opencontainers.image.title="PutPocket GLM-5.3 A6000 compatibility-gated image" \
      org.opencontainers.image.description="SM86-built current vLLM GLM5Next plus PutPocket GLM-5.3 overlay; official GLM-5.3 sparse runtime intentionally blocked" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      org.opencontainers.image.revision="9cd956c7e6cf54efa366b803cafa15ec6c2df827" \
      putpocket.task_id="T20260904-001__glm53-a6000-image" \
      putpocket.target.cuda.arch="8.6" \
      putpocket.task_built_cuda.arch="8.6-only" \
      putpocket.vendor_cuda.multiarch_possible="true" \
      putpocket.deepep.included="false" \
      putpocket.bundled_fa2.included="false" \
      putpocket.non_sm86_optional_extensions.included="false" \
      putpocket.glm53.runtime_supported="false" \
      putpocket.glm53.stateful_edit_default="off" \
      putpocket.glm53.stateful_edit_semantics="full-target-prefill-then-donor-overwrite-accuracy-ablation" \
      putpocket.glm53.true_partial_prefill="false"

COPY src/putpocket_dataset_mining/glm53_a6000_image.py /opt/putpocket/src/putpocket_dataset_mining/glm53_a6000_image.py
COPY src/putpocket_dataset_mining/glm53_a6000_stateful_edit.py /opt/putpocket/src/putpocket_dataset_mining/glm53_a6000_stateful_edit.py
COPY src/putpocket_dataset_mining/glm53_stateful_proxy.py /opt/putpocket/src/putpocket_dataset_mining/glm53_stateful_proxy.py
COPY configs/experiments/schemas/glm53_a6000_stateful_edit_control.schema.json /opt/putpocket/configs/experiments/schemas/glm53_a6000_stateful_edit_control.schema.json
COPY configs/execution/glm53_a6000_compatibility_gated.example.yaml /opt/putpocket/configs/execution/glm53_a6000_compatibility_gated.example.yaml
COPY configs/models/glm53_flash_bf16_a6000_image.lock.json /opt/putpocket/configs/models/glm53_flash_bf16_a6000_image.lock.json
COPY instrumentation/vllm/glm53_a6000_stateful_edit_accuracy_ablation.py /opt/putpocket/instrumentation/vllm/glm53_a6000_stateful_edit_accuracy_ablation.py
COPY instrumentation/vllm/glm53_a6000_compatibility_gate.py /opt/putpocket/instrumentation/vllm/glm53_a6000_compatibility_gate.py
COPY scripts/glm53_a6000 /opt/putpocket/scripts/glm53_a6000
COPY patches/vllm/9cd956c7e6cf54efa366b803cafa15ec6c2df827 /opt/putpocket/patches/vllm/9cd956c7e6cf54efa366b803cafa15ec6c2df827
COPY vendor/vllm/9cd956c7e6cf54efa366b803cafa15ec6c2df827/support_snapshot.txt /opt/putpocket/vendor/vllm/9cd956c7e6cf54efa366b803cafa15ec6c2df827/support_snapshot.txt
COPY docker/glm53_sm86/Dockerfile /opt/putpocket/docker/glm53_sm86/Dockerfile
COPY docs/GLM53_A6000_COMPATIBILITY_GATED_IMAGE.md /opt/putpocket/docs/GLM53_A6000_COMPATIBILITY_GATED_IMAGE.md

ENV PYTHONPATH=/opt/putpocket/src \
    PUTPOCKET_GLM53_STATEFUL_EDIT_CONTROL= \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_ATTENTION_BACKEND=

RUN set -eux; \
    uv pip uninstall --system \
      nvidia-cutlass-dsl \
      nvidia-cutlass-dsl-libs-base \
      nvidia-cutlass-dsl-libs-core \
      nvidia-cutlass-dsl-libs-cu12; \
    chmod 0755 /opt/putpocket/scripts/glm53_a6000/*.sh /opt/putpocket/scripts/glm53_a6000/doctor.py; \
    python3 -m py_compile \
      /opt/putpocket/src/putpocket_dataset_mining/glm53_a6000_image.py \
      /opt/putpocket/src/putpocket_dataset_mining/glm53_a6000_stateful_edit.py \
      /opt/putpocket/src/putpocket_dataset_mining/glm53_stateful_proxy.py \
      /opt/putpocket/instrumentation/vllm/glm53_a6000_stateful_edit_accuracy_ablation.py \
      /opt/putpocket/instrumentation/vllm/glm53_a6000_compatibility_gate.py

ENTRYPOINT ["/opt/putpocket/scripts/glm53_a6000/entrypoint.sh"]
CMD ["doctor"]
