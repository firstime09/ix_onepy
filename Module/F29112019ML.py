from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class F2020ML:
    def SVC_Model_NScale(dtX, dtY, test_z, r_state, kernel):
        """Model Support Vector Classification"""
        X_train, X_test, y_train, y_test = train_test_split(dtX, dtY, test_size=test_z, random_state=r_state)
        # sc = StandardScaler()
        # sc.fit(X_train)
        # X_train = sc.fit_transform(X_train)
        # X_test = sc.transform(X_test)
        best_score = 0
        for C in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for gamma in [0.001, 0.01, 0.05, 0.1, 0.5, 0.8, 1]:
                clfSVC = SVC(kernel=kernel, C=C, gamma=gamma)
                clfSVC.fit(X_train, y_train)
                score = clfSVC.score(X_test, y_test)
                if score > best_score:
                    best_score = score
                    best_C = C
                    best_gamma = gamma

        clfSVC_use = SVC(kernel=kernel, C=best_C, gamma=best_gamma)
        clfSVC_use.fit(X_train, y_train)
        return(clfSVC_use)