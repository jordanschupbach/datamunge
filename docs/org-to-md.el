;; org-to-md.el --- Batch convert Org -> Markdown with Babel execution -*- lexical-binding: t; -*-

;; Usage:
;;   emacs --batch -Q -l docs/org-to-md.el -- <input.org> <output.md>
;;
;; Behavior:
;; - Optionally applies direnv environment if `direnv` is available and an
;;   `.envrc` exists in (or above) the input file directory.
;; - Executes Org Babel source blocks (best-effort; errors are captured and do
;;   not fail the export).
;; - Exports to Markdown via `ox-md`.

(setq package-enable-at-startup nil)

(require 'org)
(require 'ob)
(require 'ox-md)
(require 'seq)
(require 'json)

(setq org-confirm-babel-evaluate nil)
(setq org-export-use-babel t)

;; Keep this conservative; add languages as needed by docs.
(org-babel-do-load-languages
 'org-babel-load-languages
 '((emacs-lisp . t)
   (shell . t)))

(defun octra--ensure-parent-dir (path)
  (let ((parent (file-name-directory (expand-file-name path))))
    (unless (file-directory-p parent)
      (make-directory parent t))))

(defun octra--direnv-root (start-dir)
  "Find nearest directory at/above START-DIR that contains an .envrc."
  (let ((dir (file-name-as-directory (expand-file-name start-dir)))
        (prev nil)
        (found nil))
    (while (and dir (not (equal dir prev)) (not found))
      (when (file-exists-p (expand-file-name ".envrc" dir))
        (setq found dir))
      (setq prev dir)
      (setq dir (file-name-directory (directory-file-name dir))))
    found))

(defun octra--apply-direnv (workdir)
  "Apply `direnv export json` for WORKDIR, if possible."
  (let ((direnv (executable-find "direnv"))
        (root (octra--direnv-root workdir)))
    (when (and direnv root)
      (let ((default-directory root))
        (with-temp-buffer
          (let ((exit-code (call-process direnv nil t nil "export" "json")))
            (when (zerop exit-code)
              (goto-char (point-min))
              (condition-case _err
                  (let* ((json-object-type 'alist)
                         (json-array-type 'list)
                         (json-key-type 'string)
                         (env (json-read)))
                    (dolist (pair env)
                      (setenv (car pair) (cdr pair)))
                    (let ((path (getenv "PATH")))
                      (when path
                        (setq exec-path
                              (append (parse-colon-path path) (list exec-directory))))))
                (error nil)))))))))

(defun octra--execute-babel-best-effort ()
  "Execute all src blocks in the current Org buffer; don't fail on errors."
  (let ((errors '()))
    (org-babel-map-src-blocks (buffer-file-name)
      (condition-case err
          (org-babel-execute-src-block)
        (error
         (push (format "%s" err) errors))))
    (when errors
      (with-current-buffer (current-buffer)
        (goto-char (point-max))
        (insert "\n\n* Export notes\n\n")
        (insert "Some code blocks failed during export but were ignored:\n\n")
        (dolist (e (reverse errors))
          (insert "- " e "\n"))))))

(defun octra-org-to-md (input-org output-md)
  "Execute babel in INPUT-ORG and export it to OUTPUT-MD."
  (let ((input (expand-file-name input-org))
        (output (expand-file-name output-md)))
    (unless (file-exists-p input)
      (error "Input Org file not found: %s" input))
    (octra--ensure-parent-dir output)
    (octra--apply-direnv (file-name-directory input))
    (with-current-buffer (find-file-noselect input)
      (unwind-protect
          (progn
            (org-mode)
            (setq-local org-export-with-toc nil)
            (setq-local org-export-with-section-numbers nil)
            (octra--execute-babel-best-effort)
            (org-export-to-file 'md output nil nil nil nil))
        (kill-buffer (current-buffer))))))

(defun octra--print-usage-and-exit ()
  (princ "Usage: emacs --batch -Q -l docs/org-to-md.el -- <input.org> <output.md>\n")
  (kill-emacs 2))

(when noninteractive
  ;; After `--`, Emacs leaves remaining args in `command-line-args-left`.
  (let* ((args (seq-filter (lambda (a) (not (string= a "--"))) command-line-args-left)))
    (cond
     ((= (length args) 2)
      (let ((in (nth 0 args))
            (out (nth 1 args)))
        (octra-org-to-md in out)
        (princ (format "Wrote %s\n" out))))
     ((= (length args) 0)
      nil)
     (t
      (octra--print-usage-and-exit)))))
