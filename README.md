### Deploy

```
cf login --sso
cf target -o "CUSTOMERTIMES_UK LTD_sub-1-jk5e0ugw" -s dev
cf push RAGu-eMLLiano
cf logs RAGu-eMLLiano --recent
```


### Resources

* https://community.sap.com/t5/technology-blogs-by-sap/how-to-setup-business-application-studio-for-python-development/ba-p/13727665
* https://github.com/saschakiefer/scp_python_template
* https://github.com/SAP-samples/sap-genai-hub-with-sap-hana-cloud-vector-engine/blob/main/genAI_vectordb.ipynb
