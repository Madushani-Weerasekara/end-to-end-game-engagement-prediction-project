import sys # Any exception that is basically getting controll this sys library will automatically have that information.

def error_message_detail(error, error_detail:sys):
    _,_,exc_tb=error_detail.exc_info() # exc_info() will talk about the execution info, this will give you 3 important info, 1st two is not intreseted at all but last will give details about exc_tb
    file_name =exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name[{0}] line[{1} error_message[{2}]]"
    file_name, exc_tb.tb_lineno, str(error)
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super.__init__(error_message) # Inherit exception class
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message