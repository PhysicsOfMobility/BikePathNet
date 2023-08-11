using LoggingExtras
using Dates


"""
    setup_logger(log_file)

Setup logger, to log messages not only to the terminal, but also to the given log file 'log_file'.
"""
function setup_logger(log_file)
    date_format = "yyyy-mm-dd HH:MM:SS"
    logger_print = FormatLogger() do io, args
        println(io, "[", args.level, "] ", "$(Dates.format(now(), date_format)): ", args.message)
    end;
    
    logger_save = FormatLogger(log_file; append=true) do io, args
        println(io, "[", args.level, "] ", "$(Dates.format(now(), date_format)): ", args.message)
    end;
    
    demux_logger = TeeLogger(
        logger_print,
        logger_save,
    );
    
    global_logger(demux_logger)
end
