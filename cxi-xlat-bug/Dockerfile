# Full elbencho compilation on Alps
#
# podman build --target export --output ./ -t elbencho:v1 -f ./Dockerfile
#
# This Dockerfile is meant to build and export the elbencho binary.
# There is no need for the squashfs file.
# Optional: enroot import -x mount -o elbencho.sqsh podman://elbencho:v1

FROM registry.suse.com/bci/bci-base:15.7 AS builder

RUN zypper update -y && \
    zypper install -y cmake gcc14-c++ git libaio-devel libboost_program_options-devel libboost_system-devel libboost_thread-devel libcurl-devel libnuma-devel libuuid-devel make openssl-devel rpm-build \
                      autoconf automake libtool libconfig-devel libuv-devel fuse-devel libyaml-devel libnl-devel libsensors4-devel libjson-c-devel

RUN git clone --depth 1 --branch release/shs-13.1.0 https://github.com/HewlettPackard/shs-cassini-headers.git
RUN git clone --depth 1 --branch release/shs-13.1.0 https://github.com/HewlettPackard/shs-cxi-driver.git
RUN git clone --depth 1 --branch release/shs-13.1.0 https://github.com/HewlettPackard/shs-libcxi.git

RUN cp -r ./shs-cassini-headers/include/* /usr/include/
RUN cp -r ./shs-cassini-headers/share/* /usr/share/
RUN cp -r ./shs-cxi-driver/include/* /usr/include/

RUN cd shs-libcxi && ./autogen.sh && ./configure --prefix=/usr && make -j8 && make install

RUN git clone https://github.com/breuner/elbencho.git
RUN cd elbencho && make CC=gcc-14 CXX=g++-14  LIBFABRIC_SUPPORT=1 BUILD_LIBFABRIC=1 -j 8

#ENTRYPOINT ["elbencho/bin/elbencho"]

FROM scratch AS export
COPY --from=builder /elbencho/bin/elbencho /elbencho
