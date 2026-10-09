"""Exact, reviewable iterator-lifetime repairs for a separate author variant.

The pinned author source remains unchanged. No interaction rules, conditions,
energies or lookup tables are altered by these replacements.
"""


def repair_author(text: str) -> str:
    patches = [
        (
            "    for (mapa_dim::iterator it_dim=it->second.begin(); it_di"
            "m!=it->second.end(); ++it_dim)//for each dimer\n"
            "      if (objects.first[it_dim->first.p1]<number[it_dim->fir"
            "st.p1] ||//if either of the two prots take part\n"
            "\t  objects.first[it_dim->first.p2]<number[it_dim->first.p2])"
            "//in more dimers that they can\n"
            "\tit->second.erase(it_dim);//this dimer will not form",
            "    for (mapa_dim::iterator it_dim=it->second.begin(); it_di"
            "m!=it->second.end();)//for each dimer\n"
            "      if (objects.first[it_dim->first.p1]<number[it_dim->fir"
            "st.p1] ||//if either of the two prots take part\n"
            "\t  objects.first[it_dim->first.p2]<number[it_dim->first.p2])"
            "//in more dimers that they can\n"
            "\tit_dim=it->second.erase(it_dim);//this dimer will not form\n"
            "      else ++it_dim;",
        ),
        (
            "\tfor (std::map<Prot,double>::iterator it1=binding_energies.f"
            "irst.begin(); it1 != binding_energies.first.end(); ++it1)\n"
            "\t  if (d_equal(it1->second,min))//if the binding energies are the same\n"
            "\t    binding_energies.first.erase(it1);//we sum to the counter",
            "\tfor (std::map<Prot,double>::iterator it1=binding_energies.f"
            "irst.begin(); it1 != binding_energies.first.end();)\n"
            "\t  if (d_equal(it1->second,min))//if the binding energies are the same\n"
            "\t    it1=binding_energies.first.erase(it1);\n"
            "\t  else ++it1;",
        ),
        (
            "\tfor (std::map<Dim, double>::iterator it2=binding_energies.s"
            "econd.begin(); it2 != binding_energies.second.end(); ++it2)\n"
            "\t  if (d_equal(it2->second, min))\n"
            "\t    binding_energies.second.erase(it2);",
            "\tfor (std::map<Dim, double>::iterator it2=binding_energies.s"
            "econd.begin(); it2 != binding_energies.second.end();)\n"
            "\t  if (d_equal(it2->second, min))\n"
            "\t    it2=binding_energies.second.erase(it2);\n"
            "\t  else ++it2;",
        ),
        (
            "  for (mapa_prot::iterator it1=objects.first.begin(); it1!=o"
            "bjects.first.end(); ++it1)\n"
            "    if (it1->second==0)\n"
            "      objects.first.erase(it1);//eliminate the proteins that have 0 copies",
            "  for (mapa_prot::iterator it1=objects.first.begin(); it1!=objects.first.end();)\n"
            "    if (it1->second==0)\n"
            "      it1=objects.first.erase(it1);//eliminate the proteins that have 0 copies\n"
            "    else ++it1;",
        ),
    ]
    # Five further exact loop repairs: metabolic competition and inventories.
    patches.extend(
        [
            (
                "    for (mapa_owm::iterator it_tuple=it->second.begin(); it_"
                "tuple != it->second.end(); "
                "++it_tuple)//for each complex\n"
                "      if (it_tuple->first.met.empty()){//protein-protein complex\n"
                "\tif (objects.first[it_tuple->first.dim.p1]<number_prots[it_t"
                "uple->first.dim.p1] ||//prot1 more "
                "involved\n"
                "\t    objects.first[it_tuple->first.dim.p2]<number_prots[it_t"
                "uple->first.dim.p2])//prot2 more "
                "involved\n"
                "\t  it->second.erase(it_tuple);//this dimer will not form\n"
                "      }\n"
                "      else if (it_tuple->first.prot!=-1){//proteins and mets are involved\n"
                "\tif (it_tuple->first.dim.empty()){//protein-met complex\n"
                "\t  if (objects.first[it_tuple->first.prot]<number_prots[it_t"
                "uple->first.prot] ||//proteins "
                "more involved\n"
                "\t      met[it_tuple->first.met]<number_mets[it_tuple->first."
                "met])//met more involved\n"
                "\t    it->second.erase(it_tuple);//this P+M will not form\n"
                "\t}\n"
                "\telse{//protein breaking a d+m complex\n"
                "\t  OWM new_tuple(-1,it_tuple->first.dim,it_tuple->first.met);\n"
                "\t  if (objects.first[it_tuple->first.prot]<number_prots[it_t"
                "uple->first.prot] ||//protein more "
                "involved\n"
                "\t      objects_with_mets[new_tuple]<number_dm[new_tuple])//d+m more involved\n"
                "\t    it->second.erase(it_tuple);//this D+M will not be broken\n"
                "\t}\n"
                "      }//proteins and mets are involved\n"
                "      else{//dimer-met complex \n"
                "\tif (objects.second[it_tuple->first.dim]<number_dims[it_tupl"
                "e->first.dim] ||//dimer more "
                "involved\n"
                "\t    met[it_tuple->first.met]<number_mets[it_tuple->first.me"
                "t])//met more involved\n"
                "\t  it->second.erase(it_tuple);//this D+M will not form\n"
                "      }\n",
                "    for (mapa_owm::iterator it_tuple=it->second.begin(); it_"
                "tuple != it->second.end();  ) "
                "{//for each complex\n"
                "      auto next_tuple = std::next(it_tuple);\n"
                "      if (it_tuple->first.met.empty()){//protein-protein complex\n"
                "\tif (objects.first[it_tuple->first.dim.p1]<number_prots[it_t"
                "uple->first.dim.p1] ||//prot1 more "
                "involved\n"
                "\t    objects.first[it_tuple->first.dim.p2]<number_prots[it_t"
                "uple->first.dim.p2])//prot2 more "
                "involved\n"
                "\t  it->second.erase(it_tuple);//this dimer will not form\n"
                "      }\n"
                "      else if (it_tuple->first.prot!=-1){//proteins and mets are involved\n"
                "\tif (it_tuple->first.dim.empty()){//protein-met complex\n"
                "\t  if (objects.first[it_tuple->first.prot]<number_prots[it_t"
                "uple->first.prot] ||//proteins "
                "more involved\n"
                "\t      met[it_tuple->first.met]<number_mets[it_tuple->first."
                "met])//met more involved\n"
                "\t    it->second.erase(it_tuple);//this P+M will not form\n"
                "\t}\n"
                "\telse{//protein breaking a d+m complex\n"
                "\t  OWM new_tuple(-1,it_tuple->first.dim,it_tuple->first.met);\n"
                "\t  if (objects.first[it_tuple->first.prot]<number_prots[it_t"
                "uple->first.prot] ||//protein more "
                "involved\n"
                "\t      objects_with_mets[new_tuple]<number_dm[new_tuple])//d+m more involved\n"
                "\t    it->second.erase(it_tuple);//this D+M will not be broken\n"
                "\t}\n"
                "      }//proteins and mets are involved\n"
                "      else{//dimer-met complex \n"
                "\tif (objects.second[it_tuple->first.dim]<number_dims[it_tupl"
                "e->first.dim] ||//dimer more "
                "involved\n"
                "\t    met[it_tuple->first.met]<number_mets[it_tuple->first.me"
                "t])//met more involved\n"
                "\t  it->second.erase(it_tuple);//this D+M will not form\n"
                "      }\n"
                "      it_tuple = next_tuple;\n"
                "    }\n",
            ),
            (
                "  for (mapa_prot::iterator it_prot=objects.first.begin(); it"
                "_prot!=objects.first.end(); "
                "++it_prot)\n"
                "    if (it_prot->second==0)\n"
                "      objects.first.erase(it_prot);//eliminate the proteins that have 0 copies",
                "  for (mapa_prot::iterator it_prot=objects.first.begin(); it"
                "_prot!=objects.first.end(); )\n"
                "    if (it_prot->second==0)\n"
                "      it_prot=objects.first.erase(it_prot);//eliminate the p"
                "roteins that have 0 copies\n"
                "    else ++it_prot;",
            ),
            (
                "  for (mapa_dim::iterator it_dim=objects.second.begin(); it_"
                "dim!=objects.second.end(); "
                "++it_dim)\n"
                "    if (it_dim->second==0)\n"
                "      objects.second.erase(it_dim);//eliminate dimers that have 0 copies",
                "  for (mapa_dim::iterator it_dim=objects.second.begin(); it_"
                "dim!=objects.second.end(); )\n"
                "    if (it_dim->second==0)\n"
                "      it_dim=objects.second.erase(it_dim);//eliminate dimers that have 0 copies\n"
                "    else ++it_dim;",
            ),
            (
                "  for (mapa_met::iterator it_met=met.begin(); it_met!=met.end(); ++it_met)\n"
                "    if (it_met->second==0)\n"
                "      met.erase(it_met);//eliminate mets that have 0 copies",
                "  for (mapa_met::iterator it_met=met.begin(); it_met!=met.end(); )\n"
                "    if (it_met->second==0)\n"
                "      it_met=met.erase(it_met);//eliminate mets that have 0 copies\n"
                "    else ++it_met;",
            ),
            (
                "  for (mapa_owm::iterator it_owm=objects_with_mets.begin(); "
                "it_owm!=objects_with_mets.end(); "
                "++it_owm)\n"
                "    if (it_owm->second==0)\n"
                "      objects_with_mets.erase(it_owm);",
                "  for (mapa_owm::iterator it_owm=objects_with_mets.begin(); "
                "it_owm!=objects_with_mets.end(); )\n"
                "    if (it_owm->second==0)\n"
                "      it_owm=objects_with_mets.erase(it_owm);\n"
                "    else ++it_owm;",
            ),
        ]
    )
    for before, after in patches:
        if text.count(before) != 1:
            raise ValueError("pinned author source differs from repair contract")
        text = text.replace(before, after, 1)
    return text
