# Bash completion for watod.  Source from ~/.bashrc:
#   source /path/to/wato_world/watod_scripts/watod_completion.bash

_watod_completions() {
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"

    local commands="install up down build run bag test exec ps logs pull"
    local components="ingest perception_2d lidar_preprocessing proposal_generation tracking label_refinement open_vocab_discovery student_training all"
    local components_dev="ingest:dev perception_2d:dev lidar_preprocessing:dev proposal_generation:dev tracking:dev label_refinement:dev open_vocab_discovery:dev student_training:dev all:dev"

    case "${prev}" in
        -c|--component)
            COMPREPLY=( $(compgen -W "${components} ${components_dev}" -- "${cur}") )
            return 0
            ;;
        run|test)
            COMPREPLY=( $(compgen -W "${components}" -- "${cur}") )
            return 0
            ;;
    esac

    COMPREPLY=( $(compgen -W "${commands} -c --component -t --terminal -v --verbose -h --help" -- "${cur}") )
}
complete -F _watod_completions watod
