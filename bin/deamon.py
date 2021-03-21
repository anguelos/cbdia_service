#!/usr/bin/env python3
import fargv
import signal
import cherrypy
import daemon
import daemon.pidfile
import os
import cherrypy
from datetime import datetime


default_params = {
    "mode": ("status", "start", "restart", "stop"),
    "name": "cbws",
    "wd": "./",
    "varroot": "tmp/",
    "stdout": "{wd}/{varroot}stdout.log",
    "stderr": "{wd}/{varroot}stderr.log",
    "pidfile": "{wd}/{varroot}{name}.pid"
}


class HelloServer(object):
    def __init__(self, params):
        self.name = params.name
        self.count = 0

    @cherrypy.expose
    def index(self):
        print(f"{self.count}  {datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}: Hello!")
        self.count += 1
        return(f"{self.count:07} Hello World!")


def start(params):
    stdout = open(params.stdout, 'w')
    stderr = open(params.stderr, 'w')
    pidfile = daemon.pidfile.PIDLockFile(params.pidfile)
    try:
        pidfile.acquire()
        print("DaemonContext before")
        with daemon.DaemonContext(stdout=stdout, stderr=stderr, working_directory=params.wd):
            print("DaemonContext Starts")
            print(f"Pidfile aquired {pidfile.path} {pidfile.read_pid()}")
            cherrypy.quickstart(HelloServer(params))
            stdout.close()
            stderr.close()
            #pidfile.release()
            print(f"Exiting DaemonContext {pidfile.path} {pidfile.read_pid()}")
            print("DaemonContext Ends")
        print("DaemonContext after")
        print(f"Pidfile {pidfile.path} {pidfile.read_pid()}")
    except IOError:
        print(f"Could not start {params.name} already running as pid {pidfile.read_pid()}")


def restart(params):
    stop(params)
    start(params)


def stop(params):
    pidfile = daemon.pidfile.PIDLockFile(params.pidfile)
    if pidfile.is_locked():
        pid = pidfile.read_pid()
        os.kill(pid, signal.SIGTERM)  # Todo(anguelos) maybe SIGKILL is better?
        print(f"SIGTERM sent to {params.name}.")
    else:
        print(f"{params.name} not running.")


def status(params):
    pidfile = daemon.pidfile.PIDLockFile(params.pidfile)
    if pidfile.is_locked():
        print(f"{params.name} running as {pidfile.read_pid()}.")
    else:
        print(f"{params.name} not running.")


if __name__ == "__main__":
    params, _ = fargv.fargv(default_params)
    if params.mode == "status":
        status(params)
    elif params.mode == "start":
        start(params)
    elif params.mode == "restart":
        restart(params)
    elif params.mode == "stop":
        stop(params)
