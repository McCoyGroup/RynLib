
######################################################################
#                   COMMON FUNCTIONS
######################################################################

RYNLIB_ENTOS_PATH=""
RYNLIB_CONIFG_PATH=""
RYNLIB_IMAGE=""
RYNLIB_DOCKER_IMAGE="McCoyGroup/RynLib"
RYNLIB_SHIFTER_IMAGE="McCoyGroup/RynLib"

function rynlib_git_update() {
  local cur;

  if [[ "$RYNLIB_PATH" == "" ]]; then
    if [[ -f ~/RynLib ]]; then
      RYNLIB_PATH=~/RynLib;
    else
      echo "RYNLIB_PATH needs to be set to know where to pull from";
    fi
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

######################################################################
#                   SYSTEM-SPECIFIC FUNCTIONS
######################################################################


RYNLIB_OPT_PATTERN=":V:";

function rynlib_update_singularity() {
  local img="$RYNLIB_IMAGE";

  module load singularity

  if [[ "$img" = "" ]]; then
    img="$PWD/rynlib.sif";
  fi

  $(rynlib_git_update);

  singularity pull $RYNLIB_DOCKER_IMAGE $img
  };

RYNLIB_OPT_PATTERN=":V:";

function rynlib_update_shifter() {
  local img="$RYNLIB_IMAGE";
  if [[ "$img" = "" ]]; then
    img="$RYNLIB_SHIFTER_IMAGE";
  fi

  $(rynlib_git_update);

  shifterimg pull img;
  };

RYNLIB_OPT_PATTERN=":V:";
function rynlib_shifter() {

    local entos="$RYNLIB_ENTOS_PATH";
    local config="$RYNLIB_CONIFG_PATH";
    local img="$RYNLIB_IMAGE";
    local vols="";

    vols=$(mcoptvalue "$RYNLIB_OPT_PATTERN" "V" $@);

    if [[ "$entos" = "" ]]; then
      entos="$PWD/entos";
    fi

    if [[ "$config" = "" ]]; then
      config="$PWD/config";
    fi

    if [[ "$img" = "" ]]; then
      img="$RYNLIB_SHIFTER_IMAGE";
    fi

    if [[ "$vols" = "" ]]; then
      vols="$config type=bind,source=$config,target=/config";
    else
      vols="$vols --mount type=bind,source=$config,target=/config";
    fi

    if [[ -f "$entos" ]]; then
      vols="$vols --mount type=bind,source=$entos,target=/entos";
    fi

    echo $(shifter --volume=$vols --image=$img python3.7 /home/RynLib/CLI.py)
}

function rynlib_singularity() {

    local entos="$RYNLIB_ENTOS_PATH";
    local config="$RYNLIB_CONIFG_PATH";
    local img="$RYNLIB_IMAGE";
    local vols="";

    vols=$(mcoptvalue "$RYNLIB_OPT_PATTERN" "V" $@);

    if [[ "$entos" = "" ]]; then
      entos="$PWD/entos";
    fi

    if [[ "$config" = "" ]]; then
      config="$PWD/config";
    fi

    if [[ "$img" = "" ]]; then
      img="$PWD/rynlib.sif";
    fi

    if [[ "$vols" = "" ]]; then
      vols="$config:/config";
    else
      vols="$vols,$config:/config";
    fi

    if [[ -f "$entos" ]]; then
      vols="$vols,$entos:/entos";
    fi

    echo $(singularity run --bind $vols $img)

}

function rynlib_docker() {

    local entos="$RYNLIB_ENTOS_PATH";
    local config="$RYNLIB_CONIFG_PATH";
    local img="$RYNLIB_IMAGE";
    local vols="";

    vols=$(mcoptvalue "$RYNLIB_OPT_PATTERN" "V" $@);

    if [[ "$entos" = "" ]]; then
      entos="$PWD/entos";
    fi

    if [[ "$config" = "" ]]; then
      config="$PWD/config";
    fi

    mkdir -p $config;

    if [[ "$img" = "" ]]; then
      img="rynlib";
    fi


    if [[ "$vols" = "" ]]; then
      vols="--mount type=bind,source=$config,target=/config";
    else
      vols="$vols --mount type=bind,source=$config,target=/config";
    fi

    if [[ -f "$entos" ]]; then
      vols="$vols --mount type=bind,source=$entos,target=/entos";
    fi

    echo docker run --rm $vols -it $img

}