#!/bin/bash

# make temporary directories
mkdir -p $HOME/tmp/shared_libraries
mkdir -p $HOME/tmp/bin

# copy over needed dependency to temproary directory
cp /usr/lib/x86_64-linux-gnu/libglut.so.3 $HOME/tmp/shared_libraries/

# if not already added, adds ~/tmp/shared_libraries to $LD_LIBRARY_PATH (for library reading on teaching node)
if [ -d "$HOME/tmp/shared_libraries" ] && [[ ":$LD_LIBRARY_PATH:" != *":$HOME/tmp/shared_libraries:"* ]]; then
    echo 'export LD_LIBRARY_PATH="$HOME/tmp/shared_libraries:$LD_LIBRARY_PATH"' >> ~/.bashrc
fi

# create prime-run
# this is needed because this configures NVIDIA graphics settings for creating the display
cat > $HOME/tmp/bin/prime-run << EOF
#!/bin/bash
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __VK_LAYER_NV_optimus=NVIDIA_only
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
exec "\$@"
EOF

# makes prime-run executable
chmod +x $HOME/tmp/bin/prime-run

# if not already added, adds ~/tmp/bin to $PATH (setup for prime-run)
if [ -d "$HOME/tmp/bin" ] && [[ ":$PATH:" != *":$HOME/tmp/bin:"* ]]; then
    echo 'export PATH="$HOME/tmp/bin:$PATH"' >> ~/.bashrc
fi

# if not already added, adds /usr/local/cuda/bin to $PATH
if [ -d "/usr/local/cuda/bin" ] && [[ ":$PATH:" != *"/usr/local/cuda/bin:"* ]]; then
    echo 'export PATH="/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
fi

# if not already added, create $XAUTHORITY variable
if [ -f "$HOME/.Xauthority" ] && [[ ":$XAUTHORITY:" != *":$HOME/.Xauthority:"* ]]; then
    echo 'export XAUTHORITY="$HOME/.Xauthority"' >> ~/.bashrc
fi

# reload ~/.bashrc so reads new variables
source ~/.bashrc