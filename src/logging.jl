"""
Setup loggers. Defaults for `log_file` and `name` is `nothing`.

If `log_file` is `nothing`, we do not log to a file.
if `name` is not `nothing`, it is used to log more information.
"""
function setup_logger(log_file=nothing, name=nothing)
    date_format = "yyyy-mm-dd HH:MM:SS"
    name = isnothing(name) ? "" : " [$name]"

    log_function(io, args) = println(io, "[", args.level, "] ", "$(Dates.format(now(), date_format))$name: ", args.message)

    logger_print = FormatLogger(log_function)

    demux_logger = if isnothing(log_file)
        TeeLogger(logger_print)
    else
        logger_save = FormatLogger(log_function, log_file; append=true)
        TeeLogger(logger_print, logger_save)
    end

    global_logger(demux_logger)
end