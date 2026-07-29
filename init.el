;;; init.el --- Batch Org export for datamunge -*- lexical-binding: t; -*-

;; Usage:
;;   emacs --batch -Q -l init.el -- examples/org/dense_linear_algebra.org build/org/dense_linear_algebra.html
;;   emacs --batch -Q -l init.el -- examples/org/dense_linear_algebra.org build/org/dense_linear_algebra.md
;;
;; Behavior:
;; - Applies the nearest `.envrc` before executing Babel blocks or exporting.
;; - Disables Org Babel confirmation prompts for batch execution.
;; - Infers the export backend from the output file extension.

(setq package-enable-at-startup nil)

(defconst datamunge-repo-root
  (file-name-directory (or load-file-name buffer-file-name default-directory))
  "Repository root for this batch Org configuration.")

(setq user-emacs-directory
      (expand-file-name ".cache/emacs/" datamunge-repo-root))

(make-directory user-emacs-directory t)

(require 'org)
(require 'ob)
(require 'ox)
(require 'ox-ascii)
(require 'ox-html)
(require 'ox-latex)
(require 'ox-md)
(require 'ox-odt)
(require 'seq)

(setq org-confirm-babel-evaluate nil)
(setq org-export-use-babel t)
;; Batch exports never collide with an interactive editor session; skip Emacs's
;; lock-file dance so a stale/concurrent lock can't abort a src-block execution.
(setq create-lockfiles nil)

;; hyperref's default link style draws a visible border box around every
;; link; when a link's text happens to wrap across a page break, pdflatex
;; can render that border as a large empty box spanning the page. Colored
;; text links have no border to mis-render, so switch to those instead.
(setq org-latex-hyperref-template
      "\\hypersetup{\n colorlinks=true,\n linkcolor=blue,\n urlcolor=blue,\n pdfauthor={%a},\n pdftitle={%t},\n pdfkeywords={%k},\n pdfsubject={%d},\n pdfcreator={%c}, \n pdflang={%L}}\n")

;; Every examples/org/*.org report and the index page it links from are
;; exported flat into the same output directory, so a relative link back to
;; index.html works uniformly from any report's HTML page.
(setq org-html-preamble t)
(setq org-html-preamble-format
      '(("en" "<div id=\"datamunge-nav\"><a href=\"index.html\">&larr; All examples</a></div>")))

(defun datamunge--convert-svg-link-for-latex (output backend _info)
  "Rewrite an Org \\includesvg link in OUTPUT to a plain \\includegraphics
pointing at a converted PDF.

For a BACKEND derived from `latex', Org's default SVG handling emits
\\includesvg{...}, which relies on the LaTeX `svg' package -- itself
requiring pdflatex to run with --shell-escape plus Inkscape on PATH, neither
of which this project's PDF export provides. Convert the referenced file to
a same-directory .pdf via rsvg-convert instead -- skipping the conversion if
an up-to-date .pdf already exists -- and emit a plain \\includegraphics."
  (if (and (org-export-derived-backend-p backend 'latex)
           (string-match "\\\\includesvg\\(\\[[^]]*\\]\\)?{\\([^}]+\\)}" output))
      (let* ((opts (or (match-string 1 output) ""))
             (raw (match-string 2 output))
             (base (expand-file-name (if (string-suffix-p ".svg" raw)
                                          (substring raw 0 -4)
                                        raw)))
             (svg (concat base ".svg"))
             (pdf (concat base ".pdf"))
             (rsvg (executable-find "rsvg-convert")))
        (unless rsvg
          (error "rsvg-convert not found on PATH; cannot embed %s in a PDF export" svg))
        (unless (file-exists-p svg)
          (error "Referenced image not found: %s" svg))
        (when (or (not (file-exists-p pdf)) (file-newer-than-file-p svg pdf))
          (unless (zerop (call-process rsvg nil nil nil "-f" "pdf" "-o" pdf svg))
            (error "rsvg-convert failed to convert %s" svg)))
        (concat "\\includegraphics" opts "{" pdf "}"))
    output))

(add-to-list 'org-export-filter-link-functions #'datamunge--convert-svg-link-for-latex)

(defvar datamunge--babel-languages nil)

(defun datamunge--enable-babel-language (lang feature)
  "Enable Babel LANG when FEATURE can be loaded."
  (when (require feature nil 'noerror)
    (push (cons lang t) datamunge--babel-languages)))

(dolist (entry '((emacs-lisp . ob-emacs-lisp)
                 (shell . ob-shell)
                 (C . ob-C)
                 (python . ob-python)
                 (js . ob-js)
                 (R . ob-R)))
  (datamunge--enable-babel-language (car entry) (cdr entry)))

(org-babel-do-load-languages
 'org-babel-load-languages
 (nreverse datamunge--babel-languages))

(defun datamunge--ensure-parent-directory (path)
  "Create the parent directory for PATH if needed."
  (let ((parent (file-name-directory (expand-file-name path))))
    (unless (file-directory-p parent)
      (make-directory parent t))))

(defun datamunge--direnv-root (start-dir)
  "Find the nearest parent of START-DIR containing `.envrc`."
  (let ((dir (file-name-as-directory (expand-file-name start-dir)))
        (prev nil)
        (found nil))
    (while (and dir (not (equal dir prev)) (not found))
      (when (file-exists-p (expand-file-name ".envrc" dir))
        (setq found dir))
      (setq prev dir)
      (setq dir (file-name-directory (directory-file-name dir))))
    found))

(defun datamunge--apply-direnv-exec (workdir)
  "Apply the evaluated direnv environment for WORKDIR.
Preserve the current PATH entries so the export shell's tools remain available."
  (let ((original-path (getenv "PATH"))
        (original-dyld-library-path (getenv "DYLD_LIBRARY_PATH"))
        (original-ld-library-path (getenv "LD_LIBRARY_PATH"))
        (original-nix-ldflags (getenv "NIX_LDFLAGS"))
        (direnv (executable-find "direnv"))
        (root (datamunge--direnv-root workdir)))
    (when (and direnv root)
      (with-temp-buffer
        (let ((status (call-process direnv nil t nil "exec" root "env" "-0")))
          (when (zerop status)
            (dolist (entry (split-string (buffer-string) "\0" t))
              (let ((eq-pos (string-search "=" entry)))
                (when eq-pos
                  (setenv (substring entry 0 eq-pos)
                          (substring entry (1+ eq-pos))))))
            (let ((path (getenv "PATH")))
              (when path
                (when original-path
                  (setq path
                        (mapconcat #'identity
                                   (delete-dups
                                    (append (parse-colon-path path)
                                            (parse-colon-path original-path)))
                                   path-separator))
                  (setenv "PATH" path))
                (setq exec-path
                      (append (parse-colon-path path)
                              (list exec-directory))))
            ;; These paths belong to the outer export shell and are required
            ;; by freshly compiled Babel executables. A nested direnv must not
            ;; discard them.
            (when original-dyld-library-path
              (setenv "DYLD_LIBRARY_PATH" original-dyld-library-path))
            (when original-ld-library-path
              (setenv "LD_LIBRARY_PATH" original-ld-library-path))
            (when original-nix-ldflags
              (setenv "NIX_LDFLAGS" original-nix-ldflags)))))))))

(defun datamunge--apply-direnv-environment (workdir)
  "Apply the nearest direnv environment for WORKDIR."
  (unless (getenv "DATAMUNGE_SKIP_DIRENV")
    (datamunge--apply-direnv-exec workdir)))

(defun datamunge--backend-for-output (output)
  "Infer the Org export backend from OUTPUT."
  (pcase (downcase (or (file-name-extension output) ""))
    ((or "adoc" "ascii" "txt") 'ascii)
    ((or "htm" "html") 'html)
    ((or "md" "markdown") 'md)
    ("odt" 'odt)
    ("pdf" 'pdf)
    ("tex" 'latex)
    (_ (error "Unsupported export extension for %s" output))))

(defun datamunge--export-pdf (output)
  "Export the current Org buffer to OUTPUT as PDF."
  (unless (executable-find "pdflatex")
    (error "pdflatex was not found in PATH"))
  (let ((pdf (org-latex-export-to-pdf nil nil nil nil nil)))
    (unless (and pdf (file-exists-p pdf))
      (error "Org did not produce a PDF"))
    (unless (file-equal-p pdf output)
      (copy-file pdf output t)
      ;; org-latex-export-to-pdf always writes next to the .org source; once
      ;; copied to the requested OUTPUT, that copy is a redundant byproduct.
      (delete-file pdf))
    output))

(defun datamunge--export-current-buffer (output)
  "Export the current Org buffer to OUTPUT."
  (let ((backend (datamunge--backend-for-output output)))
    (pcase backend
      ('pdf (datamunge--export-pdf output))
      (_ (org-export-to-file backend output nil nil nil nil nil)))))

(defun datamunge-org-export-file (input-org output-path)
  "Execute Babel for INPUT-ORG and export it to OUTPUT-PATH."
  (let ((input (expand-file-name input-org))
        (output (expand-file-name output-path)))
    (unless (file-exists-p input)
      (error "Input Org file not found: %s" input))
    (datamunge--ensure-parent-directory output)
    (datamunge--apply-direnv-environment (file-name-directory input))
    (let ((default-directory (file-name-directory input)))
      (with-current-buffer (find-file-noselect input)
        (unwind-protect
            (progn
              (org-mode)
              ;; Generate every Babel result and side-effect artifact before
              ;; export filters inspect links.  In particular, the LaTeX SVG
              ;; filter must not race ahead of a block that creates its image.
              ;; Disable Babel during the subsequent export so each block runs
              ;; exactly once.
              (org-babel-execute-buffer)
              (let ((org-export-use-babel nil))
                (datamunge--export-current-buffer output)))
          (set-buffer-modified-p nil)
          (kill-buffer (current-buffer)))))))

(defun datamunge--print-usage-and-exit ()
  "Print usage for batch export and exit with a non-zero status."
  (princ
   (concat
    "Usage: emacs --batch -Q -l init.el -- <input.org> <output.{html,md,txt,tex,odt,pdf}>\n"
    "Example: emacs --batch -Q -l init.el -- examples/org/dense_linear_algebra.org build/org/dense_linear_algebra.html\n"))
  (kill-emacs 2))

(when noninteractive
  (let ((args (seq-filter (lambda (arg) (not (string= arg "--")))
                          command-line-args-left)))
    (cond
     ((null args) nil)
     ((= (length args) 2)
      (datamunge-org-export-file (nth 0 args) (nth 1 args))
      (princ (format "Wrote %s\n" (nth 1 args))))
     (t
      (datamunge--print-usage-and-exit)))))
