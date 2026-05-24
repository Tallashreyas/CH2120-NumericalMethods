#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CH2120 Assignment GUI
Home Page → Question 1 (Gauss–Legendre) → Question 2 (BVP Solver)
"""
import math
import numpy as np
from scipy.special import erf
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QGroupBox, QFormLayout, QSpinBox,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QTabWidget, QFileDialog, QMessageBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
plt.style.use("dark_background")

# Canvas class (same as before)
class MplCanvas(FigureCanvas):
    def __init__(self, width=6, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)

import sys
import numpy as np
from numpy.linalg import eigh

import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QLineEdit, QSpinBox, QFileDialog, QMessageBox, QTableWidget,
    QTableWidgetItem, QSizePolicy, QStackedWidget, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self, width=6, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

# --- Math utilities used in this page ---
def golub_welsch_jacobi(n):
    """
    Build Jacobi tridiagonal matrix for Legendre polynomials (Gauss-Legendre)
    and return eigenvalues & eigenvectors (nodes and eigenvectors).
    """
    if n < 1:
        raise ValueError("n >= 1 required")
    k = np.arange(1, n, dtype=float)
    beta = k / np.sqrt(4.0 * k * k - 1.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)
    evals, evecs = eigh(J)   # evals sorted ascending; evecs columns correspond to eigenvalues
    return evals, evecs, J

def weights_from_evecs(evecs):
    """
    Compute Gauss-Legendre weights from eigenvectors.
    For normalized eigenvectors v(:,i) the weight is w_i = 2 * (v[0,i])**2
    (first component squared times 2).
    Also compute L2 norms of each eigenvector (should be 1 for eigh).
    """
    # ensure columns correspond to eigenvectors
    # evecs shape (n, n) columns are eigenvectors.
    n = evecs.shape[0]
    norms = np.linalg.norm(evecs, axis=0)         # L2 norms of columns
    # normalize columns (eigh returns orthonormal columns typically; still we normalize to be safe)
    evecs_normed = evecs / norms[np.newaxis, :]
    first_comp = evecs_normed[0, :]
    w = 2.0 * (first_comp ** 2)
    return w, norms, first_comp

def weights_via_lagrange_moments(nodes):
    """
    Solve V^T w = m where m_k = integral_{-1}^{1} x^k dx (moments).
    V[i,j] = nodes[i]**j (columns: powers)
    """
    x = np.asarray(nodes, dtype=float)
    n = len(x)
    V = np.vander(x, N=n, increasing=True)   # columns = x^0, x^1, ...
    m = np.zeros(n, dtype=float)
    for k in range(n):
        m[k] = 0.0 if (k % 2 == 1) else 2.0/(k+1)
    # solve V^T w = m
    w = np.linalg.solve(V.T, m)
    return w
# ---------------- Matplotlib Canvas ----------------
class MplCanvas(FigureCanvas):
    def __init__(self, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

def build_A_B_planar(n):
    x_full = collocation_nodes_planar(n)  # length m = n+2
    A_full = differentiation_matrix_1st(x_full)  # already on [0,1]
    B_full = differentiation_matrix_2nd(x_full)  # already on [0,1]
    return x_full, A_full, B_full

# ---------------- Math Utilities ----------------
def golub_welsch_legendre(n):
    k = np.arange(1, n, dtype=float)
    beta = k / np.sqrt(4*k*k - 1)
    J = np.diag(beta, 1) + np.diag(beta, -1)
    evals, evecs = np.linalg.eigh(J)
    x = evals
    w = 2 * (evecs[0, :] ** 2)
    return x, w
def collocation_nodes_planar(n):
    # n interior Gauss–Legendre nodes on [0,1], then add endpoints
    x_int, _ = golub_welsch_legendre(n)      # [-1,1]
    x_int = 0.5*(x_int + 1.0)                # -> [0,1]
    x_full = np.concatenate(([0.0], x_int, [1.0]))
    return x_full

def weights_via_lagrange_moments(x):
    n = len(x)
    V = np.vander(x, N=n, increasing=True)
    m = np.zeros(n)
    for k in range(n):
        m[k] = 0 if k % 2 else 2.0/(k+1)
    w = np.linalg.solve(V.T, m)
    return w

def barycentric_weights(x):
    n = len(x)
    w = np.ones(n)
    for i in range(n):
        diff = x[i] - np.delete(x, i)
        w[i] = 1.0 / np.prod(diff)
    return w

def differentiation_matrix_1st(x):
    n = len(x)
    w = barycentric_weights(x)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (w[j] / w[i]) / (x[i] - x[j])
        D[i, i] = -np.sum(D[i, :])
    return D

def differentiation_matrix_2nd(x):
    n = len(x)
    w = barycentric_weights(x)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = (w[j]/w[i])/(x[i]-x[j])
        D[i, i] = -np.sum(D[i, :])
    D2 = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D2[i, j] = 2*D[i, j]*(D[i, i]-1.0/(x[i]-x[j]))
        D2[i, i] = -np.sum(D2[i, :])
    return D2


class Question2Page(QWidget):
    def __init__(self, stacked=None):
        super().__init__()
        self.stacked=stacked
        # layout: left = controls + tables, right = plot
        main_layout = QHBoxLayout(self)

        # LEFT pane (controls + tables)
        left = QWidget()
        left_layout = QVBoxLayout(left)

        # Controls
        box = QGroupBox("Gauss-Legendre (via Jacobi matrix)")
        form = QFormLayout()
        self.spin_n = QSpinBox(); self.spin_n.setRange(1, 64); self.spin_n.setValue(8)
        form.addRow(QLabel("n (degree/points):"), self.spin_n)
        self.btn_compute = QPushButton("Compute nodes & weights")
        self.btn_compute.clicked.connect(self.compute_all)
        form.addRow(self.btn_compute)
        box.setLayout(form)
        left_layout.addWidget(box)

        # Table: nodes & weights (GW and Lagrange) and difference
        self.tbl_nodes = QTableWidget()
        self.tbl_nodes.setMinimumHeight(260)
        left_layout.addWidget(QLabel("Nodes and weights (GW eigenvector method vs Lagrange)"))
        left_layout.addWidget(self.tbl_nodes)

        # Table: eigenvector norms & first components
        self.tbl_evec = QTableWidget()
        left_layout.addWidget(QLabel("Eigenvector norms and first components"))
        left_layout.addWidget(self.tbl_evec)

        # Export button row
        row = QHBoxLayout()
        self.btn_export_nodes = QPushButton("Export nodes/weights CSV")
        self.btn_export_evec  = QPushButton("Export eigenvectors CSV")
        row.addWidget(self.btn_export_nodes); row.addWidget(self.btn_export_evec)
        self.btn_export_nodes.clicked.connect(self.export_nodes_csv)
        self.btn_export_evec.clicked.connect(self.export_evec_csv)
        left_layout.addLayout(row)
        btn_back = QPushButton("← Back to Home")
        btn_back.setStyleSheet("font-size: 14px; padding: 6px;")
        btn_back.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        left_layout.addWidget(btn_back)
        left_layout.addStretch(1)

        # RIGHT pane (plot)
        self.canvas = MplCanvas(width=7, height=5, dpi=100)
        # add to main layout
        main_layout.addWidget(left, stretch=1)
        main_layout.addWidget(self.canvas, stretch=2)

        # storage
        self.nodes = None
        self.w_gw = None
        self.w_lagr = None
        self.evecs = None
        self.evec_norms = None
        self.evec_first = None

    # ---- core compute function ----
    def compute_all(self):
        n = int(self.spin_n.value())
        try:
            evals, evecs, J = golub_welsch_jacobi(n)
            # eigenvalues are nodes
            nodes = evals.copy()
            # eigenvectors: columns of evecs
            w_gw, norms, first_comp = weights_from_evecs(evecs)
            w_lagr = weights_via_lagrange_moments(nodes)
            # --- Map nodes and weights from [-1,1] → [0,1] ---
            nodes = 0.5 * (nodes + 1)   # shift and scale
            w_gw = 0.5 * w_gw
            w_lagr = 0.5 * w_lagr

            # store
            self.nodes = nodes
            self.w_gw = w_gw
            self.w_lagr = w_lagr
            self.evecs = evecs
            self.evec_norms = norms
            self.evec_first = first_comp

            # fill tables and plot
            self.fill_nodes_table()
            self.fill_evec_table()
            self.plot_compare()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to compute: {e}")

    def fill_nodes_table(self):
        x = self.nodes; wg = self.w_gw; wl = self.w_lagr
        n = len(x)
        self.tbl_nodes.clear()
        self.tbl_nodes.setColumnCount(4)
        self.tbl_nodes.setRowCount(n)
        self.tbl_nodes.setHorizontalHeaderLabels(["node (x_i)", "weight (GW from evecs)", "weight (Lagrange)", "diff"])
        for i in range(n):
            self.tbl_nodes.setItem(i, 0, QTableWidgetItem(f"{x[i]:.10f}"))
            self.tbl_nodes.setItem(i, 1, QTableWidgetItem(f"{wg[i]:.12e}"))
            self.tbl_nodes.setItem(i, 2, QTableWidgetItem(f"{wl[i]:.12e}"))
            self.tbl_nodes.setItem(i, 3, QTableWidgetItem(f"{(wg[i]-wl[i]):.3e}"))
        self.tbl_nodes.resizeColumnsToContents()

    def fill_evec_table(self):
        evecs = self.evecs
        norms = self.evec_norms
        first = self.evec_first
        n = evecs.shape[1]
        self.tbl_evec.clear()
        self.tbl_evec.setColumnCount(3)
        self.tbl_evec.setRowCount(n)
        self.tbl_evec.setHorizontalHeaderLabels(["eig index", "L2 norm", "first component (normalized)"])
        for i in range(n):
            self.tbl_evec.setItem(i, 0, QTableWidgetItem(str(i+1)))
            self.tbl_evec.setItem(i, 1, QTableWidgetItem(f"{norms[i]:.12e}"))
            self.tbl_evec.setItem(i, 2, QTableWidgetItem(f"{first[i]:.12e}"))
        self.tbl_evec.resizeColumnsToContents()

    def plot_compare(self):
        ax = self.canvas.ax
        ax.clear()
        x = self.nodes
        ax.plot(x, self.w_gw, 'o-', label="GW weights (evecs)", linewidth=2, markersize=5)
        ax.plot(x, self.w_lagr, 's--', label="Lagrange weights", linewidth=1.5, markersize=4)
        ax.set_xlabel("nodes (roots)")
        ax.set_ylabel("weights")
        ax.set_title(f"n = {len(x)} : weights from eigenvectors vs Lagrange")
        ax.legend()
        ax.grid(True)
        self.canvas.draw()

    # ---- export helpers ----
    def export_nodes_csv(self):
        if self.nodes is None:
            QMessageBox.warning(self, "No data", "Compute first")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save nodes & weights", "nodes_weights.csv", "CSV Files (*.csv)")
        if not path:
            return
        arr = np.vstack([self.nodes, self.w_gw, self.w_lagr, (self.w_gw - self.w_lagr)]).T
        np.savetxt(path, arr, header="node,w_gw,w_lagr,diff", delimiter=",", comments='')
        QMessageBox.information(self, "Saved", f"Saved {path}")

    def export_evec_csv(self):
        if self.evecs is None:
            QMessageBox.warning(self, "No data", "Compute first")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save eigenvectors", "eigenvectors.csv", "CSV Files (*.csv)")
        if not path:
            return
        # Save norms and first components, and optionally eigenvectors matrix
        header = "index,norm,first_component\n"
        with open(path, 'w') as f:
            f.write(header)
            for i in range(self.evecs.shape[1]):
                f.write(f"{i+1},{self.evec_norms[i]},{self.evec_first[i]}\n")
        QMessageBox.information(self, "Saved", f"Saved {path}")

import math
from scipy.special import erf
import matplotlib.pyplot as plt

class Question3Page(QWidget):
    def __init__(self, stacked=None):
        super().__init__()
        self.stacked = stacked

        main_layout = QHBoxLayout(self)

        # --- Left panel ---
        left = QWidget()
        left_layout = QVBoxLayout(left)

        box = QGroupBox("Heat Equation Solver (Spectral Collocation)")
        form = QFormLayout()

        self.spin_nint = QSpinBox()
        self.spin_nint.setRange(2, 128)
        self.spin_nint.setValue(32)
        form.addRow(QLabel("Interior collocation points (n):"), self.spin_nint)

        self.edit_eta_max = QLineEdit("6.0")
        form.addRow(QLabel("η_max (domain size):"), self.edit_eta_max)

        self.spin_nab = QSpinBox()
        self.spin_nab.setRange(2, 128)
        self.spin_nab.setRange(2,128)
        form.addRow(QLabel("n for A/B matrices:"), self.spin_nab)

        self.btn_solve = QPushButton("Solve Heat Equation (Compare with Analytical)")
        self.btn_solve.clicked.connect(self.solve_heat_equation)
        form.addRow(self.btn_solve)

        box.setLayout(form)
        left_layout.addWidget(box)

        # Export section
        self.tableA = QTableWidget()
        self.tableB = QTableWidget()
        left_layout.addWidget(QLabel("A (D2) matrix:"))
        left_layout.addWidget(self.tableA)
        left_layout.addWidget(QLabel("B (weights) matrix:"))
        left_layout.addWidget(self.tableB)

        row = QHBoxLayout()
        self.btn_expA = QPushButton("Export A")
        self.btn_expB = QPushButton("Export B")
        row.addWidget(self.btn_expA)
        row.addWidget(self.btn_expB)
        left_layout.addLayout(row)
        self.btn_expA.clicked.connect(self.export_A)
        self.btn_expB.clicked.connect(self.export_B)

        self.status = QLabel("")
        left_layout.addWidget(self.status)

        # Back button
        btn_back = QPushButton("← Back to Home")
        btn_back.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        left_layout.addWidget(btn_back)
        left_layout.addStretch(1)

        # --- Right panel with QTabWidget ---
        self.tabs = QTabWidget()
        self.tab_results = QWidget()
        self.tab_errors = QWidget()

        # --- Tab 1: results ---
        layout_results = QVBoxLayout(self.tab_results)
        self.canvas_main = MplCanvas(width=7, height=5, dpi=100)
        layout_results.addWidget(self.canvas_main)

        # --- Tab 2: absolute errors ---
        layout_errors = QVBoxLayout(self.tab_errors)
        self.tableErr = QTableWidget()
        self.canvas_err = MplCanvas(width=7, height=5, dpi=100)
        layout_errors.addWidget(QLabel("Absolute Error Table:"))
        layout_errors.addWidget(self.tableErr)
        layout_errors.addWidget(QLabel("Absolute Error Plot:"))
        layout_errors.addWidget(self.canvas_err)

        # Add both tabs
        self.tabs.addTab(self.tab_results, "Results (f vs erf)")
        self.tabs.addTab(self.tab_errors, "Absolute Errors")

        main_layout.addWidget(left, stretch=1)
        main_layout.addWidget(self.tabs, stretch=2)

        # Storage
        self.A = None
        self.B = None
        self.nodes = None

    def solve_heat_equation(self):
        try:
            # --- User inputs ---
            n_interior = int(self.spin_nint.value())
            eta_max = float(self.edit_eta_max.text())
            n_ab = int(self.spin_nab.value())

            # --- Gauss–Legendre nodes on [-1,1] ---
            xg, wg = golub_welsch_legendre(n_interior)
            eta_int = 0.5 * eta_max * (xg + 1)
            nodes_full = np.concatenate(([0.0], eta_int, [eta_max]))
            n_full = len(nodes_full)

            # --- Differentiation matrices for the heat equation ---
            D1 = differentiation_matrix_1st(nodes_full)
            D2 = D1 @ D1

            # --- Build and solve system: f'' + 2ηf' = 0 ---
            A_sys = np.zeros((n_full, n_full))
            b = np.zeros(n_full)
            for i in range(1, n_full - 1):
                eta = nodes_full[i]
                A_sys[i, :] = D2[i, :] + 2 * eta * D1[i, :]
            A_sys[0, :] = 0; A_sys[0, 0] = 1; b[0] = 0
            A_sys[-1, :] = 0; A_sys[-1, -1] = 1; b[-1] = 1

            f_num = np.linalg.solve(A_sys, b)
            f_exact = erf(nodes_full)
            err = np.abs(f_num - f_exact)

            # --- Plot main comparison ---
            ax = self.canvas_main.ax
            ax.clear()
            ax.plot(nodes_full, f_exact, label="Analytical erf(η)", linewidth=2)
            ax.plot(nodes_full, f_num, 'o--', label="Numerical", markersize=5)
            ax.set_xlabel("η"); ax.set_ylabel("f(η)")
            ax.legend(); ax.set_title("Heat Equation: Numerical vs Analytical")
            ax.grid(True)
            self.canvas_main.draw()

            # --- Absolute error table & plot ---
            self.fill_error_table(nodes_full, f_num, f_exact, err)
            ax_e = self.canvas_err.ax
            ax_e.clear()
            ax_e.plot(nodes_full, err, 'ro-', linewidth=1.5, markersize=4)
            ax_e.set_xlabel("η"); ax_e.set_ylabel("|Error|")
            ax_e.set_title("Absolute Error Distribution")
            ax_e.grid(True)
            self.canvas_err.draw()

            # --- Compute A and B exactly like in Question 1 ---
            x_full, A_full, B_full = build_A_B_planar(n_ab)
            self.A = A_full
            self.B = B_full
            self.nodes = x_full

            # --- Show A and B in tables ---
            self.fill_table(self.tableA, self.A)
            self.fill_table(self.tableB, self.B)

            # --- Status ---
            self.status.setText(
                f"Solved successfully for n={n_ab}. "
                f"max|err|={np.max(err):.2e}, L2={np.sqrt(np.mean(err**2)):.2e}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


    def fill_error_table(self, nodes, f_num, f_exact, err):
        n = len(nodes)
        self.tableErr.clear()
        self.tableErr.setRowCount(n)
        self.tableErr.setColumnCount(4)
        self.tableErr.setHorizontalHeaderLabels(["η", "Numerical", "Analytical", "|Error|"])
        for i in range(n):
            self.tableErr.setItem(i, 0, QTableWidgetItem(f"{nodes[i]:.6f}"))
            self.tableErr.setItem(i, 1, QTableWidgetItem(f"{f_num[i]:.6e}"))
            self.tableErr.setItem(i, 2, QTableWidgetItem(f"{f_exact[i]:.6e}"))
            self.tableErr.setItem(i, 3, QTableWidgetItem(f"{err[i]:.3e}"))
        self.tableErr.resizeColumnsToContents()

    def fill_table(self, table, M):
        n, m = M.shape
        table.clear()
        table.setRowCount(n)
        table.setColumnCount(m)
        for i in range(n):
            for j in range(m):
                table.setItem(i, j, QTableWidgetItem(f"{M[i,j]:.4e}"))
        table.resizeColumnsToContents()

    def export_A(self):
        if self.A is None:
            QMessageBox.warning(self, "No Data", "Compute first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save A", "A_Q3.csv", "CSV Files (*.csv)")
        if path:
            np.savetxt(path, self.A, delimiter=",")
            QMessageBox.information(self, "Saved", f"Saved {path}")

    def export_B(self):
        if self.B is None:
            QMessageBox.warning(self, "No Data", "Compute first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save B", "B_Q3.csv", "CSV Files (*.csv)")
        if path:
            np.savetxt(path, self.B, delimiter=",")
            QMessageBox.information(self, "Saved", f"Saved {path}")
# ---------------- Question 1 Page ----------------
class Question1Page(QWidget):
    def __init__(self, stacked=None):
        super().__init__()
        self.stacked=stacked

        # ---------- Main horizontal layout ----------
        main_layout = QHBoxLayout(self)

        # ---------- LEFT SIDE ----------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Input section
        box = QGroupBox("Inputs")
        form = QFormLayout()

        self.edit_range = QLineEdit("8-32")
        form.addRow(QLabel("Plot n range:"), self.edit_range)

        self.spin_n = QSpinBox()
        self.spin_n.setRange(2, 64)
        self.spin_n.setValue(32)
        form.addRow(QLabel("n for A,B matrices:"), self.spin_n)

        # Buttons
        self.btn_plot = QPushButton("Plot Weights vs Roots")
        self.btn_plot.clicked.connect(self.plot_weights_vs_roots)
        self.btn_build = QPushButton("Compute A (D1) and B (D2)")
        self.btn_build.clicked.connect(self.build_matrices)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_plot)
        btn_row.addWidget(self.btn_build)
        form.addRow(btn_row)

        box.setLayout(form)
        left_layout.addWidget(box)

        # Tables for A and B
        self.tableA = QTableWidget()
        self.tableB = QTableWidget()
        left_layout.addWidget(QLabel("Matrix A (D1):"))
        left_layout.addWidget(self.tableA)
        left_layout.addWidget(QLabel("Matrix B (D2):"))
        left_layout.addWidget(self.tableB)

        # Export buttons
        exp_row = QHBoxLayout()
        self.btn_expA = QPushButton("Export A to CSV")
        self.btn_expB = QPushButton("Export B to CSV")
        exp_row.addWidget(self.btn_expA)
        exp_row.addWidget(self.btn_expB)
        self.btn_expA.clicked.connect(self.export_A)
        self.btn_expB.clicked.connect(self.export_B)
        left_layout.addLayout(exp_row)

        self.status = QLabel("")
        left_layout.addWidget(self.status)
        btn_back = QPushButton("← Back to Home")
        btn_back.setStyleSheet("font-size: 14px; padding: 6px;")
        btn_back.clicked.connect(lambda: self.stacked.setCurrentIndex(0))
        left_layout.addWidget(btn_back)
        left_layout.addStretch(1)

        # ---------- RIGHT SIDE ----------
        self.canvas = MplCanvas(width=7, height=5, dpi=100)

        # ---------- Combine left and right ----------
        main_layout.addWidget(left_widget, stretch=1)
        main_layout.addWidget(self.canvas, stretch=2)

        # ---------- Internal storage ----------
        self.A = None
        self.B = None
        self.nodes = None

    # ------------- Functional Parts (same as before) -------------
    def plot_weights_vs_roots(self):
        try:
            txt = self.edit_range.text().strip()
            n1, n2 = map(int, txt.split('-'))
            ax = self.canvas.ax
            ax.clear()
            for n in range(n1, n2+1):
                x, w_gw = golub_welsch_legendre(n)
                w_lg = weights_via_lagrange_moments(x)
                ax.plot(x, w_gw, 'o-', markersize=3, label=f"n={n}")
            ax.set_xlabel("Roots")
            ax.set_ylabel("Weights")
            ax.legend(loc='best', fontsize=8)
            ax.set_title(f"Gauss–Legendre Weights vs Roots (n={n1}–{n2})")
            self.canvas.draw()
            self.status.setText("Plotted Gauss–Legendre weights vs roots.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def build_matrices(self):
        try:
            n = self.spin_n.value()

            # --- full collocation grid: endpoints + interior GL nodes on [0,1]
            x, A, B = build_A_B_planar(n)   # sizes: (n+2)×(n+2)

            self.nodes, self.A, self.B = x, A, B
            self.fill_table(self.tableA, A)
            self.fill_table(self.tableB, B)

            # A has a nullspace (constants), so cond is inf: don't scare the user
            self.status.setText(f"A,B computed for n={n} on [0,1] (size {A.shape[0]}×{A.shape[1]}).")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))


    def fill_table(self, table, M):
        n, m = M.shape
        table.clear()
        table.setRowCount(n)
        table.setColumnCount(m)
        for i in range(n):
            for j in range(m):
                item = QTableWidgetItem(f"{M[i,j]:.4e}")
                item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)
                table.setItem(i, j, item)
        table.resizeColumnsToContents()

    def export_A(self):
        if self.A is None:
            QMessageBox.warning(self, "Missing", "Compute A first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save A", "A_D1.csv", "CSV Files (*.csv)")
        if path:
            np.savetxt(path, self.A, delimiter=",")
            QMessageBox.information(self, "Saved", f"Saved {path}")

    def export_B(self):
        if self.B is None:
            QMessageBox.warning(self, "Missing", "Compute B first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save B", "B_D2.csv", "CSV Files (*.csv)")
        if path:
            np.savetxt(path, self.B, delimiter=",")
            QMessageBox.information(self, "Saved", f"Saved {path}")


# ---------------- Home Page ----------------
class HomePage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked = stacked_widget
        layout = QVBoxLayout(self)

        lbl = QLabel("CH2120 Assignment GUI\nSelect a Question")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setStyleSheet("font-size: 22px; font-weight: bold; margin: 20px;")
        layout.addWidget(lbl)

        # --- Add three buttons ---
        btn1 = QPushButton("Question 1(part A): Gauss–Legendre & Collocation")
        btn2 = QPushButton("Question 1(Part B) — Nodes and Weights")
        btn3 = QPushButton("Question 2: Heat Equation Solver")


        for b in (btn1, btn2, btn3):
            b.setMinimumHeight(60)
            b.setStyleSheet("font-size: 18px;")

        layout.addWidget(btn1)
        layout.addWidget(btn2)
        layout.addWidget(btn3)
        layout.addStretch()

        # --- Link buttons to pages ---
        btn1.clicked.connect(lambda: self.stacked.setCurrentIndex(1))
        btn2.clicked.connect(lambda: self.stacked.setCurrentIndex(2))
        btn3.clicked.connect(lambda: self.stacked.setCurrentIndex(3))



# ---------------- Main Window ----------------
class AssignmentMain(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CH2120 Assignment GUI")
        self.resize(1200, 800)

        self.stacked = QStackedWidget()
        self.setCentralWidget(self.stacked)

        self.home = HomePage(self.stacked)
        self.q1 = Question1Page(self.stacked)
        self.q2 = Question2Page(self.stacked)
        self.q3 = Question3Page(self.stacked)

        self.stacked.addWidget(self.home)
        self.stacked.addWidget(self.q1)
        self.stacked.addWidget(self.q2)
        self.stacked.addWidget(self.q3)
                # --- Apply professional dark-blue theme ---
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E2E;
            }

            QWidget {
                background-color: #1E1E2E;
                color: #F1F5F9;
                font-family: "Segoe UI", "Inter", sans-serif;
                font-size: 10.5pt;
            }

            QLabel {
                color: #F1F5F9;
                font-size: 11pt;
            }

            QGroupBox {
                border: 2px solid #3B3B4F;
                border-radius: 10px;
                margin-top: 10px;
                padding: 10px;
                font-weight: bold;
                color: #E0E7FF;
                background-color: #25253A;
            }

            QLineEdit, QSpinBox, QTableWidget, QTabWidget::pane {
                background-color: #27293D;
                color: #E2E8F0;
                border: 1px solid #3B3B4F;
                border-radius: 6px;
                selection-background-color: #4F46E5;
            }

            QTableWidget {
                gridline-color: #3B3B4F;
                alternate-background-color: #222338;
                selection-color: white;
                selection-background-color: #4F46E5;
            }

            QHeaderView::section {
                background-color: #334155;
                color: #F8FAFC;
                border: none;
                padding: 5px;
            }

            QPushButton {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4F46E5, stop: 1 #3B82F6
                );
                color: white;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 600;
            }

            QPushButton:hover {
                background-color: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #6366F1, stop: 1 #60A5FA
                );
            }

            QPushButton:pressed {
                background-color: #4338CA;
            }

            QTabBar::tab {
                background-color: #27293D;
                color: #E2E8F0;
                padding: 8px 16px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
            }

            QTabBar::tab:selected {
                background-color: #4F46E5;
                color: white;
            }

            QScrollBar:vertical {
                background-color: #1E1E2E;
                width: 12px;
                margin: 15px 3px 15px 3px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #4F46E5;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #6366F1;
            }
        """)

        

def main():
    app = QApplication(sys.argv)
    win = AssignmentMain()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()