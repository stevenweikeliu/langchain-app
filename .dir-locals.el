;;; .dir-locals.el

((python-mode
  . ((eval . (setq elpy-rpc-virtualenv-path 'current))
     (eglot-workspace-configuration
      . (:python
	 (:venvPath
	  "."
	  :venv
	  ".venv"
	  :analysis
	  (:diagnosticMode
	   "workspace"
	   :autoImportCompletions
	   t)))))))   
