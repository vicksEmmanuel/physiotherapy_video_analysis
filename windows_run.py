import win32service
import win32serviceutil
import subprocess

class PythonService(win32serviceutil.ServiceFramework):
    _svc_name_ = "PythonTrainingService"
    _svc_display_name_ = "Python Training Service"
    _svc_description_= "Runs a python training script"

    def __init(self, args):
        super().__init__(args)
        self.script_path = r"C:\Users\Q2094871\Documents\physiotherapy_video_analysis\testing.py"

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)

    def SvcDoRun(self):
        self.ReportServiceStatus(win32service.SERVICE_START_PENDING)
        try:
            subprocess.call(['python', self.script_path])
        except Exception as e:
            self.SvcStop()


if __name__ == '__main__':
    win32serviceutil.HandleCommandLine(PythonService)