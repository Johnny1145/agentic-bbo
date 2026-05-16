# Minimal black-box: MySQL 5.7 + sysbench + Flask API. Build from this directory.
FROM mysql:5.7

ENV DEBIAN_FRONTEND=noninteractive \
    MYSQL_ROOT_PASSWORD=123456 \
    MYSQL_DATABASE=sbtest

RUN if command -v apt-get >/dev/null 2>&1; then \
      apt-get update && apt-get install -y --no-install-recommends \
        sysbench \
        python3 \
        python3-pip \
        ca-certificates \
        git \
        make \
        automake \
        libtool \
        pkg-config \
        autoconf \
        g++ \
        libssl-dev; \
      git clone --depth 1 --branch 1.0.20 https://github.com/akopytov/sysbench.git /tmp/sysbench; \
      cd /tmp/sysbench; \
      ./autogen.sh; \
      ./configure --without-pgsql; \
      make -j"$(nproc)"; \
      make install; \
      cd /; \
      rm -rf /tmp/sysbench; \
      rm -rf /var/lib/apt/lists/*; \
    elif command -v yum >/dev/null 2>&1; then \
      yum install -y ca-certificates || true; \
      yum install -y \
        python3 \
        python3-pip \
        git \
        make \
        automake \
        libtool \
        pkgconfig \
        autoconf \
        gcc \
        gcc-c++ \
        openssl-devel \
        mysql-community-devel \
        libaio-devel || yum install -y \
        python36 \
        python36-pip \
        git \
        make \
        automake \
        libtool \
        pkgconfig \
        autoconf \
        gcc \
        gcc-c++ \
        openssl-devel \
        mysql-community-devel \
        libaio-devel; \
      yum clean all; \
      if [ ! -x /usr/bin/python3 ] && [ -x /usr/bin/python36 ]; then ln -s /usr/bin/python36 /usr/bin/python3; fi; \
      if [ ! -x /usr/bin/pip3 ] && [ -x /usr/bin/pip3.6 ]; then ln -s /usr/bin/pip3.6 /usr/bin/pip3; fi; \
      git clone --depth 1 --branch 1.0.20 https://github.com/akopytov/sysbench.git /tmp/sysbench; \
      cd /tmp/sysbench; \
      ./autogen.sh; \
      ./configure --without-pgsql; \
      make -j"$(nproc)"; \
      make install; \
      cd /; \
      rm -rf /tmp/sysbench; \
    elif command -v microdnf >/dev/null 2>&1; then \
      microdnf install -y \
        ca-certificates \
        python3 \
        python3-pip \
        git \
        make \
        automake \
        libtool \
        pkgconf \
        autoconf \
        gcc \
        gcc-c++ \
        openssl-devel; \
      git clone --depth 1 --branch 1.0.20 https://github.com/akopytov/sysbench.git /tmp/sysbench; \
      cd /tmp/sysbench; \
      ./autogen.sh; \
      ./configure --without-pgsql; \
      make -j"$(nproc)"; \
      make install; \
      cd /; \
      rm -rf /tmp/sysbench; \
      microdnf clean all; \
    else \
      echo "No supported package manager found in base image." >&2; \
      exit 1; \
    fi \
  && pip3 install --no-cache-dir flask

WORKDIR /app
COPY server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/app/entrypoint.sh"]
