
######################################################################
#                   COMMON FUNCTIONS
######################################################################

# These three are proper environment variables that I expect people to set
#RYNLIB_ENTOS_PATH=""
#RYNLIB_CONIFG_PATH=""
#RYNLIB_IMAGE=""
#RYNLIB_CONTAINER_RUNNER="docker" # this is to allow for podman support in docker-type envs

# This was introduced literally just to make it possible to use ifort -_-
#RYNLIB_EXTENSION_PATH=""

# These three are probably never going to be changed, unless we want to change something about how
#  we're distributing the image

RYNLIB_IMAGE_NAME="rynimg"
RYNLIB_DOCKER_IMAGE="mccoygroup/rynlib:$RYNLIB_IMAGE_NAME"
RYNLIB_SINGULARITY_EXTENSION="-centos"
#RYNLIB_SHIFTER_IMAGE="registry.services.nersc.gov/b3m2a1/$RYNLIB_IMAGE_NAME:latest"
RYNLIB_SHIFTER_IMAGE="docker:$RYNLIB_DOCKER_IMAGE"

function rynlib_git_update() {
  local cur;

  if [[ "$RYNLIB_PATH" == "" ]]; then
    if [[ -d ~/RynLib ]]; then
      RYNLIB_PATH=~/RynLib;
    fi
  fi

  if [[ "$RYNLIB_PATH" == "" ]]; then
    echo "RYNLIB_PATH needs to be set to know where to pull from";
  else
    cur=$PWD;
    cd $RYNLIB_PATH;
    git pull;
    cd $cur;
  fi
}

function mcoptvalue {

  local flag_pat;
  local value_pat;
  local opt;
  local opt_string;
  local opt_whitespace;
  local OPTARG;
  local OPTIND;

  flag_pat="$1";
  shift
  value_pat="$1";
  shift

  while getopts ":$flag_pat:" opt; do
    case "$opt" in
      $value_pat)
        if [ "$opt_string" != "" ]
          then opt_whitespace=" ";
          else opt_whitespace="";
        fi;
        if [ "$OPTARG" == "" ]
          then OPTARG=true;
        fi
        opt_string="$opt_string$opt_whitespace$OPTARG"
        ;;
    esac;
  done

  OPTIND=1;

  if [ "$opt_string" == "" ]; then
    while getopts "$flag_pat" opt; do
      case "$opt" in
        $value_pat)
          if [ "$opt_string" != "" ]
            then opt_whitespace=" ";
            else opt_whitespace="";
          fi;
          OPTARG=true;
          opt_string="$opt_string$opt_whitespace$OPTARG"
          ;;
      esac;
    done
  fi

  echo $opt_string

}

function mcargcount {
  local arg;
  local arg_count=0;

  for arg in "$@"; do
    if [[ "${arg:0:2}" == "--" ]]; then
      break
    else
      arg_count=$((arg_count+1))
    fi
  done

  echo $arg_count;

}

function extract_entos {

  local img;
  local cid;
  local out;

  img=$(mcoptvalue ":e:" "e" $@);
  if [[ "$img" == "" ]]; then
    img="entos";
  fi

  out="$1";
  if [[ "$out" == "-e" ]]; then
    out="$3";
  fi
  if [[ "$out" == "" ]]; then
    out=$PWD;
  fi

  cid=$(docker run -d --entrypoint=touch $img);
  docker cp $cid:/opt/entos $out;
  docker rm $cid;

  echo "Extracted Entos to $out";

}

######################################################################
#                   SYSTEM-SPECIFIC FUNCTIONS
######################################################################

function rynlib_update_singularity() {
  local img="$RYNLIB_IMAGE";

  module load singularity

  if [[ "$img" = "" ]]; then
    img="$PWD/$RYNLIB_IMAGE_NAME.sif";
  fi

  rynlib_git_update;

  singularity pull $img docker://$RYNLIB_DOCKER_IMAGE$RYNLIB_SINGULARITY_EXTENSION

  };

function rynlib_update_shifter() {
  local img="$RYNLIB_IMAGE";

  if [[ "$img" = "" ]]; then
    img="$RYNLIB_SHIFTER_IMAGE";
  fi

  rynlib_git_update;

  shifterimg pull $img;
  };

RYNLIB_OPT_PATTERN=":eV:n:L:M:W:E:F:";
function rynlib_shifter() {

    local entos="$RYNLIB_ENTOS_PATH";
    local ext="$RYNLIB_EXTENSION_PATH";
    local config="$RYNLIB_CONFIG_PATH";
    local img="$RYNLIB_IMAGE";
    local vols="";
    local do_echo="";
    local mpi="";
    local arg_count;
    local python_version="python3";
    local entrypoint;
    local lib;
    local prof;
    local env_file;

    arg_count=$(mcargcount $@)
    vols=$(mcoptvalue $RYNLIB_OPT_PATTERN "V" ${@:1:arg_count})
    do_echo=$(mcoptvalue $RYNLIB_OPT_PATTERN "e" ${@:1:arg_count})
    mpi=$(mcoptvalue $RYNLIB_OPT_PATTERN "n" ${@:1:arg_count})
    lib=$(mcoptvalue $RYNLIB_OPT_PATTERN "L" ${@:1:arg_count})
    wdir=$(mcoptvalue $RYNLIB_OPT_PATTERN "W" ${@:1:arg_count})
    enter=$(mcoptvalue $RYNLIB_OPT_PATTERN "E" ${@:1:arg_count})
    prof=$(mcoptvalue $RYNLIB_OPT_PATTERN "M" ${@:1:arg_count})
    env_file=$(mcoptvalue $RYNLIB_OPT_PATTERN "F" ${@:1:arg_count})
    if [[ "$vols" != "" ]]; then shift 2; fi
    if [[ "$do_echo" != "" ]]; then shift; fi
    if [[ "$mpi" != "" ]]; then
      shift 2;
      local escaped="+";
      local real=" --";
      mpi=${mpi//$escaped/$real}
      escaped="="
      real=" "
      mpi=${mpi//$escaped/$real}
    fi
    if [[ "$lib" != "" ]]; then shift 2; fi
    if [[ "$wdir" != "" ]]; then shift 2; fi
    if [[ "$enter" != "" ]]; then shift 2; fi
    if [[ "$prof" != "" ]]; then shift 2; fi
    if [[ "$env_file" != "" ]]; then shift 2; fi

    if [[ "$entos" = "" ]]; then
      entos="$PWD/entos";
    fi

    if [[ "$config" = "" ]]; then
      config="$PWD/config";
    fi

    if [[ "$lib" = "" ]]; then
      lib="$RYNLIB_PATH";
    fi

    if [[ "$img" = "" ]]; then
      img="$RYNLIB_SHIFTER_IMAGE";
    fi
    img="--image=$img";

    if [[ "$vols" == "" ]]; then
      vols="$config:/config";
    else
      vols="$vols;$config:/config";
    fi
    if [[ -d "$entos" ]]; then
      if [[ "$LD_LIBRARY_PATH" == "*entos_local*" ]]; then
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
      else
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/entos/lib:/usr/entos_local/lib
      fi
      export LD_PRELOAD=libtbbmalloc_proxy.so.2
      # check if we passed the path to an entire sandbox
      if [[ -d "$entos/opt/entos" ]]; then
         vols="$vols,$entos/opt/entos:/opt/entos,$entos/usr/local:/usr/entos_local";
      else
         # otherwise just mount to opt/entos
         vols="$vols,$entos:/opt/entos";
      fi
    fi
    if [[ -d "$ext" ]]; then
      vols="$vols,$ext:/ext";
    fi
    if [[ -d "$lib" ]]; then
      vols="$vols,$lib:/home/RynLib"
    fi
    local escaped=",";
    local real=" --volume=";
    vols=${vols//$escaped/$real}
    vols="--volume=$vols";

    # Set the entrypoint and define any args we need to pass
    cmd="shifter $img $vols"
    if [[ "$wdir" != "" ]]; then
      cmd="$cmd --workdir=$wdir"
    fi
    if [[ "$prof" != "" ]]; then
      enter="mprof run $python_version /home/RynLib/CLI.py"
      cmd2="$cmd2 $vols mprof plot --output=$prof"
    elif [[ "$mpi" != "" ]]; then
      #Set the working directory
      enter="/usr/lib/mpi/bin/mpirun $python_version /home/RynLib/CLI.py"
      call="-n $mpi $call"
    elif [[ "$enter" == "" ]]; then
      enter="$python_version /home/RynLib/CLI.py"
    fi
    cmd="$cmd $enter"

    #We might want to just echo the command
    if [[ "$do_echo" == "" ]]; then
      $cmd $call $@
      if [[ "$cmd2" != "" ]]; then
        $cmd2
      fi
    else
      echo "$cmd $call"
      if [[ "$cmd2" != "" ]]; then
        echo "&&$cmd2"
      fi
    fi
}

function rynlib_singularity() {

    local entos="$RYNLIB_ENTOS_PATH";
    local ext="$RYNLIB_EXTENSION_PATH";
    local config="$RYNLIB_CONFIG_PATH";
    local img="$RYNLIB_IMAGE";
    local vols="";
    local do_echo="";
    local mpi="";
    local prof="";
    local lib="";
    local wdir="";
    local enter="";
    local cmd="";
    local cmd2="";
    local env_file;

    arg_count=$(mcargcount $@)
    vols=$(mcoptvalue $RYNLIB_OPT_PATTERN "V" ${@:1:arg_count})
    do_echo=$(mcoptvalue $RYNLIB_OPT_PATTERN "e" ${@:1:arg_count})
    mpi=$(mcoptvalue $RYNLIB_OPT_PATTERN "n" ${@:1:arg_count})
    lib=$(mcoptvalue $RYNLIB_OPT_PATTERN "L" ${@:1:arg_count})
    wdir=$(mcoptvalue $RYNLIB_OPT_PATTERN "W" ${@:1:arg_count})
    enter=$(mcoptvalue $RYNLIB_OPT_PATTERN "E" ${@:1:arg_count})
    prof=$(mcoptvalue $RYNLIB_OPT_PATTERN "M" ${@:1:arg_count})
    env_file=$(mcoptvalue $RYNLIB_OPT_PATTERN "F" ${@:1:arg_count})
    if [[ "$vols" != "" ]]; then shift 2; fi
    if [[ "$do_echo" != "" ]]; then shift; fi
    if [[ "$mpi" != "" ]]; then
      shift 2;
      local escaped="+";
      local real=" --";
      mpi=${mpi//$escaped/$real}
      escaped="="
      real=" "
      mpi=${mpi//$escaped/$real}
    fi
    if [[ "$lib" != "" ]]; then shift 2; fi
    if [[ "$wdir" != "" ]]; then shift 2; fi
    if [[ "$enter" != "" ]]; then shift 2; fi
    if [[ "$prof" != "" ]]; then shift 2; fi
    if [[ "$env_file" != "" ]]; then shift 2; fi

    if [[ "$entos" = "" ]]; then
      entos="$PWD/opt/entos";
    fi

    if [[ "$config" = "" ]]; then
      config="$PWD/config";
    fi

    if [[ "$lib" = "" ]]; then
      lib="$RYNLIB_PATH";
    fi

    if [[ "$img" = "" ]]; then
      img="$PWD/$RYNLIB_IMAGE_NAME.sif";
    fi

    if [[ "$vols" = "" ]]; then
      vols="$config:/config";
    else
      vols="$vols,$config:/config";
    fi

    export SINGULARITYENV_LD_PRELOAD=libtbbmalloc_proxy.so.2
    if [[ -d "$entos" ]]; then
      # check if we passed the path to an entire sandbox
      if [[ -d "$entos/opt/entos" ]]; then
         if [[ "$LD_LIBRARY_PATH" == "*entos_local*" ]]; then
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
         else
           export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/entos/lib:/usr/entos_local/lib
         fi
         export LD_PRELOAD=libtbbmalloc_proxy.so.2
         vols="$vols,$entos/opt/entos:/opt/entos,$entos/usr/local:/usr/entos_local";
      else
         # otherwise just mount to opt/entos
         vols="$vols,$entos:/opt/entos";
      fi
    fi
    if [[ -d "$ext" ]]; then
      vols="$vols,$ext:/ext";
    fi
    if [[ -d "$lib" ]]; then
      vols="$vols,$lib:/home/RynLib"
    fi

    # Set the entrypoint and define any args we need to pass
    cmd="singularity exec"
    if [[ "$enter" == "" ]]; then
      call="python3 /home/RynLib/CLI.py"
      if [[ "$prof" != "" ]]; then
        enter="mprof run"
        cmd2="singularity exec"
        if [[ "$wdir" != "" ]]; then
          cmd2="$cmd2 -W $wdir"
        fi
        cmd2="$cmd2 --bind $vols $img mprof plot --output=$prof"
      elif [[ "$mpi" != "" ]]; then
        #Set the working directory
        enter="/usr/lib/mpi/bin/mpirun"
        if [[ "$wdir" != "" ]]; then
          cmd="$cmd -W $wdir"
        fi
        call="-n $mpi $call"
      elif [[ "$lib" == "" ]]; then
        cmd="singularity run"
        call=""
      fi
    else
      call=""
    fi

    if [[ "$wdir" != "" ]]; then
      cmd="$cmd -W $wdir"
    fi
    cmd="$cmd --bind $vols"

    if [[ "enter" != "" ]]; then
      enter="$enter "
    fi
    #We might want to just echo the command
    if [[ "$do_echo" == "" ]]; then
      $cmd $img $enter$call $@
      if [[ "$cmd2" != "" ]]; then
        $cmd2
      fi
    else
      echo "$cmd $img $enter$call"
      if [[ "$cmd2" != "" ]]; then
        echo "&&$cmd2"
      fi
    fi
}

function rynlib_docker() {

    local entos="$RYNLIB_ENTOS_PATH";
    local ext="$RYNLIB_EXTENSION_PATH";
    local config="$RYNLIB_CONFIG_PATH";
    local img="$RYNLIB_IMAGE";
    local runner="$RYNLIB_CONTAINER_RUNNER";
    local arg_count=0;
    local vols="";
    local do_echo="";
    local mpi="";
    local prof="";
    local lib="";
    local wdir="";
    local enter="";
    local cmd="";
    local cmd2="";
    local env_file;

    arg_count=$(mcargcount $@)
    vols=$(mcoptvalue $RYNLIB_OPT_PATTERN "V" ${@:1:arg_count})
    do_echo=$(mcoptvalue $RYNLIB_OPT_PATTERN "e" ${@:1:arg_count})
    mpi=$(mcoptvalue $RYNLIB_OPT_PATTERN "n" ${@:1:arg_count})
    lib=$(mcoptvalue $RYNLIB_OPT_PATTERN "L" ${@:1:arg_count})
    wdir=$(mcoptvalue $RYNLIB_OPT_PATTERN "W" ${@:1:arg_count})
    enter=$(mcoptvalue $RYNLIB_OPT_PATTERN "E" ${@:1:arg_count})
    prof=$(mcoptvalue $RYNLIB_OPT_PATTERN "M" ${@:1:arg_count})
    env_file=$(mcoptvalue $RYNLIB_OPT_PATTERN "F" ${@:1:arg_count})
    if [[ "$vols" != "" ]]; then shift 2; fi
    if [[ "$do_echo" != "" ]]; then shift; fi
    if [[ "$mpi" != "" ]]; then
      shift 2;
      local escaped="+";
      local real=" --";
      mpi=${mpi//$escaped/$real};
      escaped="="
      real=" "
      mpi=${mpi//$escaped/$real}
    fi
    if [[ "$lib" != "" ]]; then shift 2; fi
    if [[ "$wdir" != "" ]]; then shift 2; fi
    if [[ "$enter" != "" ]]; then shift 2; fi
    if [[ "$prof" != "" ]]; then shift 2; fi
    if [[ "$env_file" != "" ]]; then shift 2; fi

    if [[ "$entos" = "" ]]; then
      entos="$PWD/entos";
    fi

    if [[ "$config" = "" ]]; then
      config="$PWD/config";
    fi

    mkdir -p $config;

    if [[ "$lib" = "" ]]; then
      lib="$RYNLIB_PATH";
    fi

    if [[ "$img" = "" ]]; then
      img="$RYNLIB_IMAGE_NAME";
    fi

    if [[ "$vols" == "" ]]; then
      vols="--mount type=bind,source=$config,target=/config";
    else
      local escaped=",";
      local real=" --mount type=bind,source=";
      vols=${vols//$escaped/$real}
      escaped=":";
      real=",target="
      vols=${vols//$escaped/$real}
      vols="--mount type=bind,source=$vols --mount type=bind,source=$config,target=/config";
    fi

    if [[ -d "$entos" ]]; then
      # check if we passed the path to an entire sandbox
      if [[ -d "$entos/opt/entos" ]]; then
         vols="$vols --mount type=bind,source=$entos/opt/entos,target=/opt/entos --mount type=bind,source=$entos/usr/local,target=/usr/entos_local";
      else
         # otherwise just mount to opt/entos
         vols="$vols --mount type=bind,source=$entos,target=/opt/entos";
      fi
    fi
    if [[ -d "$ext" ]]; then
      vols="$vols --mount type=bind,source=$ext,target=/ext";
    fi
    if [[ -d "$lib" ]]; then
      vols="$vols --mount type=bind,source=$lib,target=/home/RynLib";
    fi

    if [[ "$runner" == "" ]]; then
      runner="docker"
    fi
    # Set the entrypoint and define any args we need to pass
    cmd="$runner run --rm $vols -it"
    if [[ "$env_file" != "" ]];  then
      cmd="$cmd --env-file=$env_file"
    fi
    if [[ "$enter" == "" ]]; then
      call="python3 /home/RynLib/CLI.py"
      # if we want to profile our job, we really want to do 2 docker calls at once
      if [[ "$prof" != "" ]]; then
        cmd="$cmd --entrypoint=mprof"
        call="run $call"
        if [[ "$mpi" != "" ]]; then
          call="/usr/lib/mpi/bin/mpirun -n $mpi $call"
        fi
        cmd2="$runner run --entrypoint=mprof"
        if [[ "$wdir" != "" ]]; then
          cmd2="$cmd2 -w=$wdir"
        fi
        cmd2="$cmd2 $vols $img plot --output=$prof"
      elif [[ "$mpi" != "" ]]; then
        cmd="$cmd --entrypoint=/usr/lib/mpi/bin/mpirun"
        call="-n $mpi $call"
      else
        call=""
      fi
    else
      cmd="$cmd --entrypoint=$enter"
    fi

    #Set the working directory
    if [[ "$wdir" != "" ]]; then
      cmd="$cmd -w=$wdir"
    fi

    #We might want to just echo the command
    if [[ "$do_echo" == "" ]]; then
      $cmd $img $call $@
      if [[ "$cmd2" != "" ]]; then
        $cmd2
      fi
    else
      echo "$cmd $img $call"
      if [[ "$cmd2" != "" ]]; then
        echo "&&$cmd2"
      fi
    fi
}

function rynlib() {
  local img="$RYNLIB_IMAGE";
  local cmd;

  # if shifter exists at all...
  if [[ -x "/usr/bin/shifter" ]]; then
    cmd="rynlib_shifter"
  fi

  # if our image is already tagged as a .sif
  if [[ "$img" == *".sif" ]] && [[ "$cmd" == "" ]]; then
    cmd="rynlib_singularity";
  fi

  # if we've got a .sif file we can run
  if [[ "$img" = "" ]] && [[ -f "$PWD/$RYNLIB_IMAGE_NAME.sif" ]] && [[ "$cmd" == "" ]]; then
    cmd="rynlib_singularity";
  fi

  if [[ "$cmd" == "" ]]; then
    cmd="rynlib_docker";
  fi

  $cmd $@;

}