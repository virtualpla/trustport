FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

WORKDIR /opt/trustport

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY trustport ./trustport
COPY configs ./configs
RUN pip install --no-cache-dir --no-deps -e .

ENTRYPOINT ["trustport"]
CMD ["--help"]
