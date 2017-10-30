import glog
import os
import datetime

def get_logger(logPath='./logs', fileName=None, logLevel="INFO", is_Del=True):
    
    surName = datetime.datetime.now().strftime('%m-%d-%y_%H-%M-%S')
    if fileName is None:
        fileName = surName
    else:
        fileName = fileName + '-' + surName

    logFile = "{0}/{1}.log".format(logPath, fileName)
    
    is_Exist_Path = os.path.exists(logPath)
    is_Exist_File = os.path.exists(logFile)
    if not is_Exist_Path:
        os.system('mkdir ' + logPath)
    if is_Del and is_Exist_File:
        os.system('rm ' + logFile)
        
    handler = glog.logging.FileHandler(logFile)
    glog.logger.addHandler(handler)
    glog.setLevel(logLevel)

    glog.info("Happy Logging!")
    return glog, logFile
