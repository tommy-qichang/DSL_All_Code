import ctypes
import logging
import threading
import traceback

from ..message import Message


class MPIReceiveThread(threading.Thread):
    def __init__(self, comm, rank, size, name, q):
        super(MPIReceiveThread, self).__init__()
        self._stop_event = threading.Event()
        self.comm = comm
        self.rank = rank
        self.size = size
        self.name = name
        self.q = q
        self.daemon = True

    def run(self):
        logging.debug("Starting RThread:" + self.name + ". Process ID = " + str(self.rank))
        while not self.stopped():
            try:
                msg_str = self.comm.recv()
                msg = Message()
                msg.init(msg_str)
                self.q.put(msg)
            except Exception:
                logging.debug("Exception RThread:" + self.name + ". Process ID = " + str(self.rank))
                traceback.print_exc()
                break
        logging.debug("Ending RThread:" + self.name + ". Process ID = " + str(self.rank))

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    def get_id(self):
        # returns id of the respective thread
        if hasattr(self, '_thread_id'):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id),
                                                         ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), 0)
            raise SystemError("PyThreadState_SetAsyncExc failed")
