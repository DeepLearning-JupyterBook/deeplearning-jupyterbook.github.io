# Working with Servers

In a real deep learning project, we often need to run our code on a server computer with powerful 
GPUs.

## Connection

We can use the Secure Shell Protocol (**SSH**) to connect to a server computer:

    ssh <USER_NAME>@<SERVER_IP>

After entering the password, you have made a connection to the server computer. You have access to
the terminal of the server computer where you can execute different commands.

## Jupyter

You can start a `jupyter notebook/lab` on the server and open it on your local browser.

On the server run:

    jupyter notebook --no-browser --port=<REMOTE_PORT>

This is similar to how we start a `jupyter notebook/lab` on a local machine with an extra argument of
`--no-browser`  that starts the notebook without opening a browser.

Note that the port `<REMOTE_PORT>` you selected might not be the one that gets assigned to you 
(e.g., in case it’s already being used).

Once `jupyter notebook` has started, it will show you an URL with a security token that you will 
need to open the notebook in your local browser.

On your local terminal:

    ssh -L <LOCAL_PORT>:localhost:<REMOTE_PORT> <REMOTE_USER>@<SERVER_IP>

This command links the `<REMOTE_PORT>` to the specified `<LOCAL_PORT>`.

Once the connection is set up, you can open `jupyter notebook` in your browser by entering: 

    http://localhost:<LOCAL_PORT>/

You may be asked to enter a token (see above).

## TensorBoard

Similar to Jupyter, you can start the `tensorboard` on the server and open it on your local
browser. There are two options. 

###  "No particular address" placeholder

On the server run:

    tensorboard --host 0.0.0.0 --logdir <LOG_DIR> --port <REMOTE_PORT>

On the browser of your local computer:

    http://<SERVER_IP>:<REMOTE_PORT>/

Some servers' firewall block this method. If this is the case, use the local port forwarding. 

### Local port forwarding

On the server run:

    tensorboard --logdir <LOG_DIR> --port <REMOTE_PORT>

On local terminal:

    ssh -L <LOCAL_PORT>:localhost:<REMOTE_PORT> <REMOTE_USER>@<SERVER_IP>

On the browser of your local computer:

    http://localhost:<LOCAL_PORT>/

## Tmux

In most use-case scenarios, you want:
* to run a script for several hours,
* to run several scripts at the same time.

`tmux` facilitates this functionality.
* To create a new session run `tmux new -s <SESSION_NAME>`.
* To detach from a session press `ctr+b` and then `d`.
* To attach to a session press `tmux a -t <SESSION_NAME>`.
* To easily move across sessions press `ctr+b` and then `s`.

Online resources:
* Do [Tumux tutorial](https://leimao.github.io/blog/Tmux-Tutorial/) to learn more.
* Check the [cheat sheet](https://tmuxcheatsheet.com/) to learn more about the shortcuts.