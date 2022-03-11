import tkinter as tk
from tkinter import ttk
from ttkthemes import ThemedTk
from tkinter import messagebox as mb

# ML Stuff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class PredictGUI:
    def __init__(self):
        # self.window = tk.Tk()
        self.window = ThemedTk(theme="arc")

        self.var_op_cd_sexe = tk.StringVar()
        self.var_mtrev = tk.IntVar()
        self.var_nb_enf = tk.IntVar()
        self.var_sit_fam = tk.StringVar()
        self.var_cd_tmt = tk.StringVar()
        self.var_cat_cl = tk.StringVar()
        self.var_age_ad = tk.IntVar()

        self.lr_model = LogisticRegression()

        self.train()

    def train(self):
        users_array = np.genfromtxt('previsions.csv', delimiter=',')
        users_array = np.delete(users_array, (0), axis=0)
        average_drop_rate = (np.mean([i[0] for i in users_array]) * 100)

        X_train, X_test, y_train, y_test = train_test_split([i[1:] for i in users_array], [i[0] for i in users_array],
                                                            test_size=0.2, random_state=42)
        # np.where(np.isnan(users_array))

        self.lr_model.fit(X_train, y_train)

    def predict(self):
        cd_sex = self.var_op_cd_sexe.get()
        mt_rev = self.var_mtrev.get()
        nb_enf = self.var_nb_enf.get()
        age_ad = self.var_age_ad.get()

        cd_tmt_0, cd_tmt_2, cd_tmt_4, cd_tmt_6 = (0, ) * 4
        cd_tmt = self.var_cd_tmt.get()
        if cd_tmt == "0":
            cd_tmt_0 = 1
        elif cd_tmt == "2":
            cd_tmt_2 = 1
        elif cd_tmt == "4":
            cd_tmt_4 = 1
        elif cd_tmt == "6":
            cd_tmt_6 = 1

        cd_sit_fam_A, cd_sit_fam_B, cd_sit_fam_C, cd_sit_fam_D, cd_sit_fam_E, cd_sit_fam_F, cd_sit_fam_G, cd_sit_fam_M, \
        cd_sit_fam_P, cd_sit_fam_S, cd_sit_fam_U, cd_sit_fam_V = (0, ) * 12
        cd_sit_fam = self.var_sit_fam.get()

        if cd_sit_fam == "A":
            cd_sit_fam_A = 1
        elif cd_sit_fam == "B":
            cd_sit_fam_B = 1
        elif cd_sit_fam == "C":
            cd_sit_fam_C = 1
        elif cd_sit_fam == "D":
            cd_sit_fam_D = 1
        elif cd_sit_fam == "E":
            cd_sit_fam_E = 1
        elif cd_sit_fam == "F":
            cd_sit_fam_F = 1
        elif cd_sit_fam == "G":
            cd_sit_fam_G = 1
        elif cd_sit_fam == "M":
            cd_sit_fam_M = 1
        elif cd_sit_fam == "P":
            cd_sit_fam_P = 1
        elif cd_sit_fam == "S":
            cd_sit_fam_S = 1

        cd_cat_cl_10, cd_cat_cl_21, cd_cat_cl_22, cd_cat_cl_23, cd_cat_cl_24, cd_cat_cl_25, cd_cat_cl_32, cd_cat_cl_40, \
        cd_cat_cl_50, cd_cat_cl_61, cd_cat_cl_82, cd_cat_cl_98 = (0, ) * 12
        cd_cat_cl = self.var_cat_cl.get()

        if cd_cat_cl == "10":
            cd_cat_cl_10 = 1
        elif cd_cat_cl == "21":
            cd_cat_cl_21 = 1
        elif cd_cat_cl == "22":
            cd_cat_cl_22 = 1
        elif cd_cat_cl == "23":
            cd_cat_cl_23 = 1
        elif cd_cat_cl == "24":
            cd_cat_cl_24 = 1
        elif cd_cat_cl == "25":
            cd_cat_cl_25 = 1
        elif cd_cat_cl == "32":
            cd_cat_cl_32 = 1
        elif cd_cat_cl == "40":
            cd_cat_cl_40 = 1
        elif cd_cat_cl == "50":
            cd_cat_cl_50 = 1
        elif cd_cat_cl == "61":
            cd_cat_cl_61 = 1
        elif cd_cat_cl == "82":
            cd_cat_cl_82 = 1
        elif cd_cat_cl == "98":
            cd_cat_cl_98 = 1

        new_client = [
            [cd_sex, mt_rev, nb_enf, age_ad, cd_tmt_0, cd_tmt_2, cd_tmt_4, cd_tmt_6, cd_sit_fam_A, cd_sit_fam_B,
             cd_sit_fam_C,
             cd_sit_fam_D, cd_sit_fam_E, cd_sit_fam_F, cd_sit_fam_G, cd_sit_fam_M, cd_sit_fam_P, cd_sit_fam_S,
             cd_sit_fam_U,
             cd_sit_fam_V, cd_tmt_0, cd_tmt_2, cd_tmt_4, cd_tmt_6, cd_cat_cl_10, cd_cat_cl_21, cd_cat_cl_22,
             cd_cat_cl_23,
             cd_cat_cl_24, cd_cat_cl_25, cd_cat_cl_32, cd_cat_cl_40, cd_cat_cl_50, cd_cat_cl_61, cd_cat_cl_82,
             cd_cat_cl_98]]

        Y_pred = self.lr_model.predict_proba(new_client)
        drop_probability = Y_pred[0][1] * 100

        print(f"""CDSEXE: {self.var_op_cd_sexe.get()}
MTREV: {self.var_mtrev.get()}
NBENF: {self.var_nb_enf.get()}
CDSITFAM: {self.var_sit_fam.get()}
CDTMT: {self.var_cd_tmt.get()}
CDCATCL: {self.var_cat_cl.get()}
AGEAD: {self.var_age_ad.get()}
""")
        print(f"Drop prob : {drop_probability}")

        mb.showinfo("Prévision client", f"La probabilité de départ pour le client précisé est de {drop_probability:.3f} %")

    def build(self):
        self.window.title("Client DropOut Predictor")
        self.window.rowconfigure(0, minsize=800, weight=1)
        self.window.columnconfigure(1, minsize=800, weight=1)

        # txt_edit = tk.Text(window)
        fr_buttons = tk.Frame(self.window, relief=tk.RAISED, bd=1)

        btn_predict = ttk.Button(fr_buttons, text="Predict", command=self.predict)

        listeOptions = ['linear', 'linear', 'linear 2', 'linear: origins', 'Yet Another Linear']
        mode_var = tk.StringVar(self.window)
        mode_var.set(listeOptions[0])
        om = ttk.OptionMenu(fr_buttons, mode_var, *listeOptions)

        btn_predict.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        om.grid(row=2, column=0, sticky="ew", padx=5)

        fr_buttons.grid(row=0, column=0, sticky="ns")

        fr_form = tk.Frame(self.window, bd=3)

        # dem,CDSEXE,MTREV,NBENF,CDSITFAM,CDTMT,CDCATCL,AGEAD

        # CDSEXE

        fr_sexe = tk.Frame(fr_form, relief=tk.RAISED, bd=1, pady=6)
        var_lbl_cd_sexe = tk.StringVar()
        var_lbl_cd_sexe.set("CDSEXE")
        lbl_cd_sexe = tk.Label(fr_sexe, textvariable=var_lbl_cd_sexe)
        lbl_cd_sexe.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        cd_sexe_options = ('1', '1', '2', '3')
        self.var_op_cd_sexe.set(cd_sexe_options[0])
        om_cd_sexe = ttk.OptionMenu(fr_sexe, self.var_op_cd_sexe, *cd_sexe_options)
        om_cd_sexe.grid(row=1, column=0, sticky="ew", padx=5)

        fr_sexe.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # MTREV

        fr_mtrev = tk.Frame(fr_form, relief=tk.RAISED, bd=1, pady=6)
        var_lbl_cd_mtrev = tk.StringVar()
        var_lbl_cd_mtrev.set("MTREV")
        lbl_cd_mtrev = tk.Label(fr_mtrev, textvariable=var_lbl_cd_mtrev)
        lbl_cd_mtrev.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        sb_mtrev = ttk.Spinbox(fr_mtrev, from_=0, to=100_000_000, increment=100, textvariable=self.var_mtrev)
        sb_mtrev.grid(row=1, column=0, sticky="ew", padx=5)

        fr_mtrev.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        # NBENF

        fr_nbenf = tk.Frame(fr_form, relief=tk.RAISED, bd=1, pady=6)
        var_lbl_cd_nbenf = tk.StringVar()
        var_lbl_cd_nbenf.set("Nombre d'enfants")
        lbl_cd_nbenf = tk.Label(fr_nbenf, textvariable=var_lbl_cd_nbenf)
        lbl_cd_nbenf.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        sb_nbenf = ttk.Spinbox(fr_nbenf, from_=0, to=100, increment=1, textvariable=self.var_nb_enf)
        sb_nbenf.grid(row=1, column=0, sticky="ew", padx=5)

        fr_nbenf.grid(row=0, column=2, sticky="ew", padx=5, pady=5)

        # CDSITFAM
        fr_sit_fam = tk.Frame(fr_form, relief=tk.RAISED, bd=1, pady=6)
        var_lbl_cd_sit_fam = tk.StringVar()
        var_lbl_cd_sit_fam.set("Situation Familial")
        lbl_cd_sit_fam = tk.Label(fr_sit_fam, textvariable=var_lbl_cd_sit_fam)
        lbl_cd_sit_fam.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        cd_sit_fam_options = ('A', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'P', 'S', 'U', 'V')
        self.var_sit_fam.set(cd_sit_fam_options[0])
        om_cd_sit_fam = ttk.OptionMenu(fr_sit_fam, self.var_sit_fam, *cd_sit_fam_options)
        om_cd_sit_fam.grid(row=1, column=0, sticky="ew", padx=5)

        fr_sit_fam.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        # CDTMT
        fr_cd_tmt = tk.Frame(fr_form, relief=tk.RAISED, bd=1, pady=6)
        var_lbl_cd_cd_tmt = tk.StringVar()
        var_lbl_cd_cd_tmt.set("CD TMT")
        lbl_cd_cd_tmt = tk.Label(fr_cd_tmt, textvariable=var_lbl_cd_cd_tmt)
        lbl_cd_cd_tmt.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        cd_tmt_options = ('0', '0', '2', '6', '4')
        self.var_cd_tmt.set(cd_tmt_options[0])
        om_cd_cd_tmt = ttk.OptionMenu(fr_cd_tmt, self.var_cd_tmt, *cd_tmt_options)
        om_cd_cd_tmt.grid(row=1, column=0, sticky="ew", padx=5)

        fr_cd_tmt.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # CDCATCL
        fr_cd_cat_cl = tk.Frame(fr_form, relief=tk.RAISED, bd=1, pady=6)
        var_lbl_cd_cat_cl = tk.StringVar()
        var_lbl_cd_cat_cl.set("Categories Client")
        lbl_cd_cat_cl = tk.Label(fr_cd_cat_cl, textvariable=var_lbl_cd_cat_cl)
        lbl_cd_cat_cl.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        cd_cat_cl_options = ('10', '10', '21', '22', '23', '24', '25', '32', '40', '50', '61', '82', '98')
        self.var_cat_cl.set(cd_cat_cl_options[0])
        om_cd_cat_cl = ttk.OptionMenu(fr_cd_cat_cl, self.var_cat_cl, *cd_cat_cl_options)
        om_cd_cat_cl.grid(row=1, column=0, sticky="ew", padx=5)

        fr_cd_cat_cl.grid(row=1, column=2, sticky="ew", padx=5, pady=5)

        # AGEAD
        fr_age_ad = tk.Frame(fr_form, relief=tk.RAISED, bd=1, pady=6)
        var_lbl_cd_age_ad = tk.StringVar()
        var_lbl_cd_age_ad.set("Age Admission")
        lbl_cd_age_ad = tk.Label(fr_age_ad, textvariable=var_lbl_cd_age_ad)
        lbl_cd_age_ad.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        sb_age_ad = ttk.Spinbox(fr_age_ad, from_=0, to=90, increment=1, textvariable=self.var_age_ad)
        sb_age_ad.grid(row=1, column=0, sticky="ew", padx=5)

        fr_age_ad.grid(row=2, column=0, sticky="ew", padx=5, pady=5)

        # End

        fr_form.grid(row=0, column=1, sticky="nsew")

    def show(self):
        self.window.mainloop()


if __name__ == '__main__':
    gui = PredictGUI()
    gui.build()
    gui.show()
