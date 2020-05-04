
######################################################################
#                   COMMON FUNCTIONS
######################################################################

# These three are proper environment variables that I expect people to set
#RYNLIB_ENTOS_PATH=""
#RYNLIB_CONIFG_PATH=""
#RYNLIB_IMAGE=""

# These three are probably never going to be changed, unless we want to change something about how
#  we're distributing the image
RYNLIB_IMAGE_NAME="rynimg"
RYNLIB_DOCKER_IMAGE="mccoygroup/rynlib:$RYNLIB_IMAGE_NAME"
RYNLIB_SHIFTER_IMAGE="registry.services.nersc.gov/b3m2a1/$RYNLIB_IMAGE_NAME"

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
  docker cp $cid:/entos $out;
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

  singularity pull $img docker://$RYNLIB_DOCKER_IMAGE-centos
  };

function rynlib_update_shifter() {
  local img="$RYNLIB_IMAGE";
  if [[ "$img" = "" ]]; then
    img="$RYNLIB_SHIFTER_IMAGE:latest";
  fi

  rynlib_git_update;

  shifterimg pull img;
  };

RYNLIB_OPT_PATTERN=":eV:";
function rynlib_shifter() {

    local entos="$RYNLIB_ENTOS_PATH";
    local config="$RYNLIB_CONFIG_PATH";
    local img="$RYNLIB_IMAGE";
    local vols="";
    local do_echo="";

    vols=$(mcoptvalue "$RYNLIB_OPT_PATTERN" "V" $@);
    do_echo=$(mcoptvalue "$RYNLIB_OPT_PATTERN" "e" $@);

    if [[ "$entos" = "" ]]; then
      entos="$PWD/entos";
    fi

    if [[ "$config" = "" ]]; then
      config="$PWD/config";
    fi

    if [[ "$img" = "" ]]; then
      img="$RYNLIB_SHIFTER_IMAGE:latest";
    fi

    if [[ "$vols" = "" ]]; then
      vols="$config type=bind,source=$config,target=/config";
    else
      vols="$vols --mount type=bind,source=$config,target=/config";
    fi

    if [[ -d "$entos" ]]; then
      vols="$vols --mount type=bind,source=$entos,target=/entos";
    fi

    if [[ "$do_echo" = "" ]]; then
      shifter --volume=$vols --image=$img python3.7 /home/RynLib/CLI.py $@
    else
      echo "shifter --volume=$vols --image=$img python3.7 /home/RynLib/CLI.py $@"
    fi
}

function rynlib_singularity() {

    local entos="$RYNLIB_ENTOS_PATH";
    local config="$RYNLIB_CONFIG_PATH";
    local img="$RYNLIB_IMAGE";
    local vols="";
    local do_echo="";

    vols=$(mcoptvalue "$RYNLIB_OPT_PATTERN" "V" $@);
    do_echo=$(mcoptvalue "$RYNLIB_OPT_PATTERN" "e" $@);

    vols=$(mcoptvalue "$RYNLIB_OPT_PATTERN" "V" $@);

    if [[ "$entos" = "" ]]; then
      entos="$PWD/entos";
    fi

    if [[ "$config" = "" ]]; then
      config="$PWD/config";
    fi

    if [[ "$img" = "" ]]; then
      img="$PWD/$RYNLIB_IMAGE_NAME.sif";
    fi

    if [[ "$vols" = "" ]]; then
      vols="$config:/config";
    else
      vols="$vols,$config:/config";
    fi

    if [[ -d "$entos" ]]; then
      vols="$vols,$entos:/entos";
    fi

    if [[ "$do_echo" == "" ]]; then
      singularity run --bind $vols $img $@
    else
      echo "singularity run --bind $vols $img $@"
    fi
}

function rynlib_docker() {

    local entos="$RYNLIB_ENTOS_PATH";
    local config="$RYNLIB_CONFIG_PATH";
    local img="$RYNLIB_IMAGE";
    local vols="";
    local do_echo="";

    vols=$(mcoptvalue "$RYNLIB_OPT_PATTERN" "V" $@);
    do_echo=$(mcoptvalue "$RYNLIB_OPT_PATTERN" "e" $@);

    vols=$(mcoptvalue "$RYNLIB_OPT_PATTERN" "V" $@);

    if [[ "$entos" = "" ]]; then
      entos="$PWD/entos";
    fi

    if [[ "$config" = "" ]]; then
      config="$PWD/config";
    fi

    mkdir -p $config;

    if [[ "$img" = "" ]]; then
      img="$RYNLIB_IMAGE_NAME";
    fi


    if [[ "$vols" = "" ]]; then
      vols="--mount type=bind,source=$config,target=/config";
    else
      vols="$vols --mount type=bind,source=$config,target=/config";
    fi

    if [[ -d "$entos" ]]; then
      vols="$vols --mount type=bind,source=$entos,target=/entos";
    fi

    if [[ "$do_echo" == "" ]]; then
      docker run --rm $vols -it $img $@
    else
      echo "docker run --rm $vols -it $img $@"
    fi
}

function rynlib() {
  local img="$RYNLIB_IMAGE";
  local cmd;

  if [[ -x "shifter" ]]; then
    cmd="rynlib_shifter"
  fi


  if [[ "$img" == "*.sif" ]]; then
    if [[ "$cmd" == "" ]]; then
      cmd="rynlib_singularity";
    fi
  fi

  if [[ "$img" = "" ]] && [[ -f "$PWD/$RYNLIB_IMAGE_NAME.sif" ]]; then
    if [[ "$cmd" == "" ]]; then
      cmd="rynlib_singularity";
    fi
  fi

  if [[ "$cmd" == "" ]]; then
    cmd="rynlib_docker";
  fi

  $cmd $@;

}