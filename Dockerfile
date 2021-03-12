FROM osrf/ros:noetic-desktop-full
# FROM arm64v8/ros:noetic   # choose this image if you are working on a MacBook M1

WORKDIR /root/ws/
COPY . .

# INSTALL ROS AND OTHER DEPENDENCIES
RUN apt-get update && apt-get install -y \
  ros-${ROS_DISTRO}-ros-tutorials \
  ros-${ROS_DISTRO}-common-tutorials \
  python3-pip \
  flake8 && \
  rm -rf /var/lib/apt/lists/*

# INSTALL PYTHON PACKAGES
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# EXTEND BASHRC
RUN echo "source /root/ws/devel_isolated/setup.bash" >> ~/.bashrc
