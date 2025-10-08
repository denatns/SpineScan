FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

ENV PORT=8501

# ==============================
# SYSTEM SETUP
# ==============================
RUN pip install --upgrade pip
WORKDIR /app

RUN apt clean && \
    apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 libmagic1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install opencv-python

# ==============================
# DEPENDENCIES
# ==============================
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# ==============================
# CUSTOM MODULE OVERRIDE
# ==============================
COPY ./src/plotting.py /opt/conda/lib/python3.7/site-packages/ultralytics/utils/plotting.py

# ==============================
# APP SETUP
# ==============================
COPY . /app
EXPOSE ${PORT}

# ==============================
# RUN STREAMLIT
# ==============================
CMD streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0
