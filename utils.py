import numpy as np

def change(test, perc, y_pred, model, idx):
    testX_ = test.copy()
    testX_[:, idx, :] *= 1+perc/100.
    y_pred_ = model.predict(testX_).reshape(-1)
    print(round(np.median((y_pred_ - y_pred)/y_pred)*100.,2))


# In[23]:


def blinding(test, perc, y_pred, model, idx_on, idx_off):
    testX_ = test.copy()[:10]
    testX_[:, idx_off, :] = [np.array([0.0]) for i in range(10)]
    testX_[:, idx_on, :] = [np.array([i/10.]) for i in range(10)]
    print(
        round(
            np.mean(
                    (model.predict(testX_) - testX_[:,idx_on,:])**2.
                    ),
            3)
    )
