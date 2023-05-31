# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import streamlit as st
    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier

    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target
    clf = RandomForestClassifier()
    clf.fit(X, Y)
    st.title("Iris Flower Prediction App")
    st.header("Enter the parameters below to predict the type of iris flower:")
    sepal_length = st.slider("Sepal length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
    sepal_width = st.slider("Sepal width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
    petal_length = st.slider("Petal length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
    petal_width = st.slider("Petal width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))
    prediction_button = st.button("Predict")

    if prediction_button:
        input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
        prediction = clf.predict(input_data)
        predicted_species = iris.target_names[prediction[0]]
        st.write("Predicted species:", predicted_species)
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
