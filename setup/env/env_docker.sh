
######################################################################
#                   COMMON FUNCTIONS (copy of env_common.sh)
######################################################################

RYNLIB_ENTOS_PATH=""
RYNLIB_CONIFG_PATH=""
RYNLIB_IMAGE=""

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
#                   DOCKER-SPECIFIC FUNCTIONS
######################################################################


RYNLIB_OPT_PATTERN=":V:";
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
      img="rynimg";
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